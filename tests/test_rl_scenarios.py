"""Tests for training scenario panels.

A fixed panel's whole value is that "identical training conditions" is checkable, so it must reconstruct a wildfireGP
run's landscapes exactly --- not approximately.
"""

import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from wildfireGP.network import NodeState, create_grid  # noqa: E402
from wildfireGP.rl.env import EnvConfig, WildfireEnv  # noqa: E402
from wildfireGP.rl.scenarios import RESERVED_SEEDS, load_gp_panel  # noqa: E402


def write_config(tmp_path, **overrides):
    scenario = {
        "rows": 30,
        "cols": 30,
        "seed": 5,
        "landscapes": 2,
        "ignition_nodes_per_landscape": [[[10, 10], [10, 11]], [[20, 20], [21, 20]]],
        "treatments_per_step": 3,
        "max_steps": 40,
        "intervention_delay": 3,
        "min_treatment_distance": 0,
        "terrain_smoothing": 3.0,
        "fuel_smoothing": 3.0,
        "water_fraction": 0.0,
        "rock_fraction": 0.0,
        "wind_per_landscape": [{"speed": 12.0, "direction": 90.0}, {"speed": 48.0, "direction": 200.0}],
        "fuel_moisture": 0.2,
    }
    scenario.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"scenario": scenario}))
    return path


def test_load_panel_reads_seeds_wind_and_ignitions(tmp_path):
    panel = load_gp_panel(write_config(tmp_path))
    assert len(panel) == 2
    assert [s.landscape_seed for s in panel.scenarios] == [5, 6], "land_seed must be base seed + index"
    assert [s.wind_speed for s in panel.scenarios] == [12.0, 48.0]
    assert panel.scenarios[0].ignition_nodes == ((10, 10), (10, 11))
    assert panel.max_steps == 40


def test_load_panel_rejects_unreproducible_and_inconsistent_configs(tmp_path):
    with pytest.raises(ValueError, match="seed=None"):
        load_gp_panel(write_config(tmp_path, seed=None))
    with pytest.raises(ValueError, match="inconsistent"):
        load_gp_panel(write_config(tmp_path, landscapes=3))


def test_reserved_seeds_cover_the_canonical_convention():
    """Training 0-3, validation 42, test 99 --- per-episode sampling must avoid all of them."""
    assert RESERVED_SEEDS == {0, 1, 2, 3, 42, 99}


def test_config_from_panel_inherits_simulation_parameters(tmp_path):
    panel = load_gp_panel(write_config(tmp_path))
    cfg = EnvConfig.from_panel(panel)
    assert (cfg.rows, cfg.cols, cfg.max_steps) == (30, 30, 40)
    assert cfg.intervention_delay == 3 and cfg.min_treatment_distance == 0
    assert cfg.panel is panel


# -- Fixed panel -----------------------------------------------------------------


def test_panel_episodes_use_only_panel_scenarios(tmp_path):
    panel = load_gp_panel(write_config(tmp_path))
    env = WildfireEnv(EnvConfig.from_panel(panel))
    allowed = {(s.landscape_seed, s.wind_speed) for s in panel.scenarios}
    for i in range(12):
        _, info = env.reset(seed=i)
        assert (info["landscape_seed"], info["wind_speed"]) in allowed


def test_panel_covers_every_scenario(tmp_path):
    panel = load_gp_panel(write_config(tmp_path))
    env = WildfireEnv(EnvConfig.from_panel(panel))
    seen = {env.reset(seed=i)[1]["landscape_seed"] for i in range(40)}
    assert seen == {s.landscape_seed for s in panel.scenarios}


def test_panel_uses_recorded_ignition_not_the_fizzle_filter(tmp_path):
    """The recorded clusters were already validated by the original run; re-searching would pick different cells."""
    panel = load_gp_panel(write_config(tmp_path))
    env = WildfireEnv(EnvConfig.from_panel(panel))
    by_seed = {s.landscape_seed: s.ignition_nodes for s in panel.scenarios}
    for i in range(10):
        _, info = env.reset(seed=i)
        assert tuple(env.ignition_nodes) == by_seed[info["landscape_seed"]]


def test_panel_landscape_matches_create_grid_with_the_recorded_seed(tmp_path):
    """The reconstruction claim: same seed and parameters must give back the identical terrain."""
    panel = load_gp_panel(write_config(tmp_path))
    env = WildfireEnv(EnvConfig.from_panel(panel))
    _, info = env.reset(seed=0)
    expected = create_grid(
        rows=panel.rows,
        cols=panel.cols,
        terrain_smoothing=panel.terrain_smoothing,
        fuel_smoothing=panel.fuel_smoothing,
        water_fraction=panel.water_fraction,
        rock_fraction=panel.rock_fraction,
        seed=info["landscape_seed"],
    )
    np.testing.assert_array_equal(env.state.elevation, expected.elevation)
    np.testing.assert_array_equal(env.state.slope, expected.slope)
    # fuel differs only where the recorded ignition burned it
    untouched = env.state.state == NodeState.UNBURNED
    np.testing.assert_array_equal(env.state.fuel[untouched], expected.fuel[untouched])


# -- Per-episode sampling -----------------------------------------------------------------


def test_wind_speed_choices_are_drawn_from_the_given_set():
    cfg = EnvConfig(rows=25, cols=25, max_steps=30, min_burned=20, wind_speed_choices=(12.0, 48.0))
    env = WildfireEnv(cfg)
    speeds = {env.reset(seed=100 + i)[1]["wind_speed"] for i in range(15)}
    assert speeds <= {12.0, 48.0}
    assert len(speeds) == 2, "expected both choices to appear"


def test_random_wind_direction_varies_across_episodes_and_is_fixed_within_one():
    cfg = EnvConfig(rows=25, cols=25, max_steps=30, min_burned=20, wind_direction_random=True)
    env = WildfireEnv(cfg)
    directions = []
    for i in range(6):
        env.reset(seed=200 + i)
        start = env.state.wind_direction
        for _ in range(4):
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert env.state.wind_direction == start, "wind must not drift during an episode"
        directions.append(start)
    assert len(set(directions)) > 1
    assert all(0.0 <= d < 360.0 for d in directions)


def test_free_sampling_draws_a_fresh_landscape_each_episode():
    cfg = EnvConfig(rows=25, cols=25, max_steps=30, min_burned=20)
    env = WildfireEnv(cfg)
    seeds = {env.reset(seed=300 + i)[1]["landscape_seed"] for i in range(8)}
    assert len(seeds) == 8
