"""
Tests for scripts/run_gp.py save helpers.

The GP evolutionary loop itself is integration-tested by running the full script end-to-end via
main(); the unit tests here cover the output serialisation helpers in isolation.
"""

import json
import pathlib
import pickle

import dill
import numpy as np
import pytest
from deap import tools

from scripts.run_gp import (
    _make_output_dir,
    _save_config,
    _save_final_population_dill,
    _save_hof,
    _save_population,
    _save_stats,
    main,
)
from wildfireGP.gp import GPConfig, _register_types, build_toolbox
from wildfireGP.network import (
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def out_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path


@pytest.fixture()
def scenario() -> dict:
    return {
        "rows": 10,
        "cols": 10,
        "seed": 0,
        "ignition_nodes": [[5, 5], [5, 6], [6, 5]],
        "treatments_per_step": 2,
        "max_steps": 20,
        "wind_speed": 20.0,
        "wind_direction": 0.0,
        "fuel_moisture": 0.2,
    }


@pytest.fixture()
def config() -> GPConfig:
    return GPConfig(population_size=4, generations=2)


@pytest.fixture()
def toolbox_and_pop(config):
    _register_types()
    rng = np.random.default_rng(0)
    graph = create_grid(10, 10, seed=0)
    set_wind(graph, speed=20.0, direction=0.0)
    set_fuel_moisture(graph, moisture=0.2)
    ignition = select_ignition_node(graph, rng)
    tb = build_toolbox([(graph, [ignition])], 2, 20, rng, config)
    pop = tb.population(n=config.population_size)
    for ind in pop:
        ind.fitness.values = (10.0,)
        ind.peak_burning = 5
    return tb, pop


@pytest.fixture()
def hof(toolbox_and_pop):
    _, pop = toolbox_and_pop
    h = tools.HallOfFame(len(pop))
    h.update(pop)
    return h


# ---------------------------------------------------------------------------
# _save_config
# ---------------------------------------------------------------------------


def test_save_config_gp_config_round_trips(out_dir, config, scenario):
    _save_config(out_dir, config, scenario)
    data = json.loads((out_dir / "config.json").read_text())
    assert data["gp_config"]["population_size"] == config.population_size
    assert data["gp_config"]["generations"] == config.generations


def test_save_config_scenario_round_trips(out_dir, config, scenario):
    _save_config(out_dir, config, scenario)
    data = json.loads((out_dir / "config.json").read_text())
    assert data["scenario"]["rows"] == scenario["rows"]
    assert data["scenario"]["seed"] == scenario["seed"]


# ---------------------------------------------------------------------------
# _save_stats
# ---------------------------------------------------------------------------


def _make_logbook() -> tools.Logbook:
    logbook = tools.Logbook()
    logbook.header = ["gen", "fitness", "size", "peak", "height"]
    logbook.record(gen=0, fitness={"avg": 8.0, "min": 5.0}, size={"avg": 7.0}, peak={"avg": 4.0}, height={"avg": 3.0})
    logbook.record(gen=1, fitness={"avg": 6.0, "min": 4.0}, size={"avg": 6.5}, peak={"avg": 3.0}, height={"avg": 2.5})
    return logbook


def test_save_stats_correct_generation_count(out_dir):
    _save_stats(out_dir, _make_logbook())
    rows = json.loads((out_dir / "stats.json").read_text())
    assert len(rows) == 2


def test_save_stats_gen_field_present(out_dir):
    _save_stats(out_dir, _make_logbook())
    rows = json.loads((out_dir / "stats.json").read_text())
    assert rows[0]["gen"] == 0
    assert rows[1]["gen"] == 1


def test_save_stats_fitness_avg_serialised(out_dir):
    _save_stats(out_dir, _make_logbook())
    rows = json.loads((out_dir / "stats.json").read_text())
    assert rows[0]["fitness_avg"] == 8.0


# ---------------------------------------------------------------------------
# _save_population
# ---------------------------------------------------------------------------


def test_save_population_round_trips_size(out_dir, toolbox_and_pop):
    _, pop = toolbox_and_pop
    _save_population(out_dir, pop, tools.Logbook())
    with open(out_dir / "population.pkl", "rb") as f:
        data = pickle.load(f)
    assert len(data["population"]) == len(pop)


# ---------------------------------------------------------------------------
# _save_final_population_dill
# ---------------------------------------------------------------------------


def test_save_final_population_dill_round_trips_count(out_dir, toolbox_and_pop):
    _, pop = toolbox_and_pop
    _save_final_population_dill(out_dir, pop)
    with open(out_dir / "final_population.dill", "rb") as f:
        entries = dill.load(f)
    assert len(entries) == len(pop)


def test_save_final_population_dill_entries_are_expr_callable_pairs(out_dir, toolbox_and_pop):
    _, pop = toolbox_and_pop
    _save_final_population_dill(out_dir, pop)
    with open(out_dir / "final_population.dill", "rb") as f:
        entries = dill.load(f)
    for expr, func in entries:
        assert isinstance(expr, str) and len(expr) > 0
        assert callable(func)


# ---------------------------------------------------------------------------
# _save_hof
# ---------------------------------------------------------------------------


def test_save_hof_dill_loads_as_callable(out_dir, hof):
    _save_hof(out_dir, hof)
    with open(out_dir / "hof_0.dill", "rb") as f:
        func = dill.load(f)
    assert callable(func)


def test_save_hof_expr_is_non_empty_string(out_dir, hof):
    _save_hof(out_dir, hof)
    expr = (out_dir / "hof_0.expr").read_text()
    assert len(expr) > 0


def test_save_hof_count_matches_hof_size(out_dir, hof):
    _save_hof(out_dir, hof)
    saved = list(out_dir.glob("hof_*.dill"))
    assert len(saved) == len(hof)


# ---------------------------------------------------------------------------
# _make_output_dir
# ---------------------------------------------------------------------------


def test_make_output_dir_uses_run_name(tmp_path):
    out = _make_output_dir(tmp_path, "my_run")
    assert out == tmp_path / "my_run"
    assert out.is_dir()


def test_make_output_dir_without_run_name_uses_timestamp(tmp_path):
    out = _make_output_dir(tmp_path)
    import re

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", out.name)
    assert out.is_dir()


# ---------------------------------------------------------------------------
# main() smoke test
# ---------------------------------------------------------------------------


def test_main_creates_expected_output_files(tmp_path):
    main(
        [
            "--pop",
            "4",
            "--gens",
            "2",
            "--hof",
            "2",
            "--seed",
            "0",
            "--rows",
            "5",
            "--cols",
            "5",
            "--results-dir",
            str(tmp_path),
        ]
    )
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    out = run_dirs[0]
    assert (out / "config.json").exists()
    assert (out / "stats.json").exists()
    assert (out / "population.pkl").exists()
    assert (out / "final_population.dill").exists()
    assert (out / "hof_0.dill").exists()
    assert (out / "hof_0.expr").exists()


def test_main_run_name_creates_named_dir(tmp_path):
    main(
        [
            "--pop",
            "4",
            "--gens",
            "2",
            "--hof",
            "1",
            "--seed",
            "0",
            "--rows",
            "5",
            "--cols",
            "5",
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "experiment_A",
        ]
    )
    assert (tmp_path / "experiment_A").is_dir()


def test_main_max_tree_height_and_nodes_in_config(tmp_path):
    main(
        [
            "--pop",
            "4",
            "--gens",
            "2",
            "--hof",
            "1",
            "--seed",
            "0",
            "--rows",
            "5",
            "--cols",
            "5",
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "size_test",
            "--max-tree-height",
            "10",
            "--max-tree-nodes",
            "128",
        ]
    )
    data = json.loads((tmp_path / "size_test" / "config.json").read_text())
    assert data["gp_config"]["max_tree_height"] == 10
    assert data["gp_config"]["max_tree_nodes"] == 128
