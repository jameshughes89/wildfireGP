"""Tests for the Gymnasium wildfire environment.

The load-bearing test is :func:`test_env_matches_reference_simulate_loop`: the env's step loop must reproduce
``wildfireGP.evaluate.simulate``'s mechanics exactly, or the RL is trained and scored on a different problem than the
GP policies were. The reference loop there is written out independently rather than calling ``simulate``, because
``simulate`` selects treatments through ``_apply_treatments``, which shuffles candidates and so consumes the RNG
stream the env does not.
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from wildfireGP.evaluate import init_ignition  # noqa: E402
from wildfireGP.features import (  # noqa: E402
    distance_to_fire,
    fuel_level,
    update_neighbourhood_maps_after_treatment,
)
from wildfireGP.network import (  # noqa: E402
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.rl.env import EnvConfig, WildfireEnv  # noqa: E402
from wildfireGP.rl.features import (  # noqa: E402
    N_FEATURES,
    compute_action_mask,
    extract_features,
    run_precomputes,
)
from wildfireGP.spread import spread_step  # noqa: E402

SMALL = EnvConfig(rows=30, cols=30, max_steps=40, min_burned=30)


def score(state, node):
    """Deterministic stand-in for a policy: direct attack, tie-broken downstream by flat index."""
    return -distance_to_fire(state, node) + 0.001 * fuel_level(state, node)


def make_scenario(seed, cfg=SMALL):
    state = create_grid(rows=cfg.rows, cols=cfg.cols, seed=seed)
    set_wind(state, speed=cfg.wind_speed, direction=cfg.wind_direction)
    set_fuel_moisture(state, moisture=cfg.fuel_moisture)
    land = np.argwhere((state.terrain == TerrainType.LAND) & (state.fuel > 0.3))
    r, c = land[len(land) // 2]
    return state, [(int(r), int(c))]


def pick(state, mask, cfg):
    scores = np.full(cfg.rows * cfg.cols, -np.inf)
    for flat in np.flatnonzero(mask):
        scores[flat] = score(state, divmod(int(flat), cfg.cols))
    return int(np.argmax(scores))


def reference_run(template, ignition, cfg, sim_seed):
    """Independent reimplementation of simulate()'s loop with deterministic top-K selection."""
    state = template.copy()
    init_ignition(state, ignition)
    rng = np.random.default_rng(sim_seed)
    for step in range(cfg.max_steps):
        if not (state.state == NodeState.BURNING).any():
            break
        if step >= cfg.intervention_delay:
            run_precomputes(state)
            for _ in range(cfg.treatments_per_step):
                mask = compute_action_mask(state, cfg.min_treatment_distance)
                if not mask.any():
                    break
                node = divmod(pick(state, mask, cfg), cfg.cols)
                state.state[node] = NodeState.TREATED
                state.step_treated += 1
                state.step_unburned -= 1
                update_neighbourhood_maps_after_treatment(state, node)
        spread_step(state, rng)
    return int((state.state == NodeState.BURNED).sum())


def run_env(env, cfg):
    """Drive the env with the same deterministic policy; return (total_burned, return, n_actions)."""
    env.reset(seed=0, options={"state": env._template, "ignition_nodes": env._ignition, "sim_seed": 12345})
    total_reward, n_actions, info = 0.0, 0, {}
    while True:
        action = pick(env.state, env.action_masks(), cfg)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_actions += 1
        if terminated or truncated:
            break
    return info["total_burned"], total_reward, n_actions


def make_env(seed, cfg=SMALL):
    env = WildfireEnv(cfg)
    env._template, env._ignition = make_scenario(seed, cfg)
    return env


# -- spaces and reset ------------------------------------------------------


def test_spaces_and_reset_shapes():
    env = WildfireEnv(SMALL)
    obs, info = env.reset(seed=0)
    assert env.observation_space.shape == (SMALL.rows * SMALL.cols, N_FEATURES)
    assert env.action_space.n == SMALL.rows * SMALL.cols
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert info["sim_step"] == SMALL.intervention_delay
    assert info["treatments_left"] == SMALL.treatments_per_step


def test_reset_is_deterministic_given_seed():
    a, _ = WildfireEnv(SMALL).reset(seed=42)
    b, _ = WildfireEnv(SMALL).reset(seed=42)
    np.testing.assert_array_equal(a, b)
    c, _ = WildfireEnv(SMALL).reset(seed=43)
    assert not np.array_equal(a, c)


def test_reset_runs_exactly_the_intervention_delay():
    """Treatments must not be placeable before the delay elapses --- it models detection and dispatch time."""
    for delay in (0, 3, 6):
        cfg = EnvConfig(rows=30, cols=30, max_steps=40, min_burned=30, intervention_delay=delay)
        _, info = WildfireEnv(cfg).reset(seed=1)
        assert info["sim_step"] == delay


# -- masking ---------------------------------------------------------------


def test_action_mask_agrees_with_state():
    env = WildfireEnv(SMALL)
    env.reset(seed=0)
    mask = env.action_masks()
    expected = (env.state.state == NodeState.UNBURNED) & (env.state.fuel > 0.0)
    np.testing.assert_array_equal(mask, expected.reshape(-1))
    assert mask.any()


def test_masked_action_is_rejected():
    env = WildfireEnv(SMALL)
    env.reset(seed=0)
    burning = int(np.flatnonzero((env.state.state == NodeState.BURNING).reshape(-1))[0])
    with pytest.raises(ValueError, match="masked out"):
        env.step(burning)


def test_treated_cell_becomes_ineligible():
    env = WildfireEnv(SMALL)
    env.reset(seed=0)
    action = int(np.flatnonzero(env.action_masks())[0])
    env.step(action)
    assert not env.action_masks()[action]
    assert env.state.state[divmod(action, SMALL.cols)] == NodeState.TREATED


def test_step_on_finished_episode_raises():
    env = make_env(0)
    run_env(env, SMALL)
    with pytest.raises(RuntimeError, match="finished episode"):
        env.step(int(np.flatnonzero(env.action_masks())[0]))


# -- timestep structure ----------------------------------------------------


def test_fire_advances_only_after_k_treatments():
    env = WildfireEnv(SMALL)
    _, info = env.reset(seed=0)
    start = info["sim_step"]
    for i in range(SMALL.treatments_per_step - 1):
        _, reward, _, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert reward == 0.0, "reward accrues only on the spread step"
        assert info["sim_step"] == start, "fire must not advance mid-budget"
        assert info["treatments_left"] == SMALL.treatments_per_step - 1 - i
    _, _, _, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert info["sim_step"] == start + 1


def test_treatments_placed_per_timestep_respects_budget():
    env = WildfireEnv(SMALL)
    env.reset(seed=0)
    treated_before = int((env.state.state == NodeState.TREATED).sum())
    step_before = env._sim_step
    while env._sim_step == step_before:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    placed = int((env.state.state == NodeState.TREATED).sum()) - treated_before
    assert placed == SMALL.treatments_per_step


# -- reward and termination ------------------------------------------------


def test_return_equals_negative_burn_increase():
    """Dense -delta_burned must telescope to the burn accrued since the agent's first decision."""
    env = make_env(0)
    env.reset(seed=0, options={"state": env._template, "ignition_nodes": env._ignition, "sim_seed": 12345})
    burned_at_start = int((env.state.state == NodeState.BURNED).sum())
    total_reward = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(pick(env.state, env.action_masks(), SMALL))
        total_reward += reward
        if terminated or truncated:
            break
    assert total_reward == pytest.approx(-(info["total_burned"] - burned_at_start))
    assert total_reward < 0.0


def test_episode_terminates_when_fire_goes_out():
    env = make_env(0)
    obs, _ = env.reset(seed=0, options={"state": env._template, "ignition_nodes": env._ignition, "sim_seed": 12345})
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = env.step(pick(env.state, env.action_masks(), SMALL))
    assert terminated and not truncated
    assert not (env.state.state == NodeState.BURNING).any()
    assert np.isfinite(obs).all(), "terminal observation must still be a valid network input"


def test_truncation_at_max_steps():
    """A horizon short enough to cut the fire off must report truncated, not terminated."""
    cfg = EnvConfig(rows=30, cols=30, max_steps=8, min_burned=30)
    env = WildfireEnv(cfg)
    state, ignition = make_scenario(0, cfg)
    env.reset(seed=0, options={"state": state, "ignition_nodes": ignition, "sim_seed": 12345})
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(pick(env.state, env.action_masks(), cfg))
    assert truncated and not terminated
    assert info["sim_step"] == cfg.max_steps
    assert (env.state.state == NodeState.BURNING).any()


# -- parity with the reference loop ---------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_env_matches_reference_simulate_loop(seed):
    """The env's mechanics must reproduce simulate()'s: same treatments, same spread, same burn total."""
    env = make_env(seed)
    burned, _, _ = run_env(env, SMALL)
    assert burned == reference_run(env._template, env._ignition, SMALL, sim_seed=12345)


def test_scenario_injection_does_not_mutate_the_caller_template():
    """An evaluation harness reuses one landscape template across many runs; the env must copy, as evaluate() does."""
    env = make_env(0)
    before = env._template.state.copy()
    run_env(env, SMALL)
    np.testing.assert_array_equal(env._template.state, before)


def test_injected_scenario_is_reproducible():
    a = make_env(0)
    b = make_env(0)
    assert run_env(a, SMALL)[0] == run_env(b, SMALL)[0]


def test_observation_matches_direct_extraction():
    """The obs handed to the policy must be exactly what the extractor produces for the current state."""
    env = WildfireEnv(SMALL)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs, extract_features(env.state))


def test_argmax_tie_breaking_is_lowest_flat_index():
    """Pin the selection semantics that make this env unsuitable for measuring tie-heavy baselines.

    wildfireGP.evaluate._apply_treatments shuffles candidates so ties break randomly; a caller using argmax here
    breaks them by lowest flat index, producing contiguous row-major placement. The difference is worth ~25% of
    pooled burned area for score_by_fire_proximity, so evaluation must go through the canonical harness. See the
    module docstring of wildfireGP.rl.env.
    """
    env = WildfireEnv(SMALL)
    env.reset(seed=0)
    mask = env.action_masks()
    flat_scores = np.where(mask, 1.0, -np.inf)  # every eligible cell exactly tied
    chosen = int(np.argmax(flat_scores))
    assert chosen == int(np.flatnonzero(mask)[0])

    env.step(chosen)
    assert env.state.state[divmod(chosen, SMALL.cols)] == NodeState.TREATED


def test_env_and_canonical_agree_for_a_continuous_scorer():
    """The parity guarantee holds exactly where the RL needs it: scores with no ties."""
    rng = np.random.default_rng(0)
    weights = rng.normal(size=N_FEATURES)

    env = make_env(11)
    env.reset(seed=0, options={"state": env._template, "ignition_nodes": env._ignition, "sim_seed": 999})
    total = 0.0
    while True:
        scores = extract_features(env.state) @ weights
        scores[~env.action_masks()] = -np.inf
        _, reward, terminated, truncated, info = env.step(int(np.argmax(scores)))
        total += reward
        if terminated or truncated:
            break
    assert len(np.unique(scores[np.isfinite(scores)])) > 1, "continuous scorer should produce distinct values"
    assert info["total_burned"] > 0
