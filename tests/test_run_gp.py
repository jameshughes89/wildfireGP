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

from scripts.run_gp import _save_config, _save_hof, _save_population, _save_stats
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
        "ignition_node": [5, 5],
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
    tb = build_toolbox(graph, [ignition], 2, 20, rng, config)
    pop = tb.population(n=config.population_size)
    for ind in pop:
        ind.fitness.values = (10.0, 5.0)
    return tb, pop


# ---------------------------------------------------------------------------
# _save_config
# ---------------------------------------------------------------------------


def test_save_config_creates_file(out_dir, config, scenario):
    _save_config(out_dir, config, scenario)
    assert (out_dir / "config.json").exists()


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
    logbook.header = ["gen", "fitness", "size"]
    logbook.record(gen=0, fitness={"avg": np.array([8.0, 4.0]), "min": np.array([5.0, 2.0])}, size={"avg": 7.0})
    logbook.record(gen=1, fitness={"avg": np.array([6.0, 3.0]), "min": np.array([4.0, 1.0])}, size={"avg": 6.5})
    return logbook


def test_save_stats_creates_file(out_dir):
    _save_stats(out_dir, _make_logbook())
    assert (out_dir / "stats.json").exists()


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
    assert rows[0]["fitness_avg"] == [8.0, 4.0]


# ---------------------------------------------------------------------------
# _save_population
# ---------------------------------------------------------------------------


def test_save_population_creates_file(out_dir, toolbox_and_pop):
    _, pop = toolbox_and_pop
    _save_population(out_dir, pop, tools.Logbook())
    assert (out_dir / "population.pkl").exists()


def test_save_population_round_trips_size(out_dir, toolbox_and_pop):
    _, pop = toolbox_and_pop
    _save_population(out_dir, pop, tools.Logbook())
    with open(out_dir / "population.pkl", "rb") as f:
        data = pickle.load(f)
    assert len(data["population"]) == len(pop)


# ---------------------------------------------------------------------------
# _save_hof
# ---------------------------------------------------------------------------


def test_save_hof_creates_dill_files(out_dir, toolbox_and_pop):
    tb, pop = toolbox_and_pop
    _save_hof(out_dir, pop, n=2, toolbox=tb)
    assert (out_dir / "hof_0.dill").exists()
    assert (out_dir / "hof_1.dill").exists()


def test_save_hof_creates_expr_files(out_dir, toolbox_and_pop):
    tb, pop = toolbox_and_pop
    _save_hof(out_dir, pop, n=2, toolbox=tb)
    assert (out_dir / "hof_0.expr").exists()
    assert (out_dir / "hof_1.expr").exists()


def test_save_hof_dill_loads_as_callable(out_dir, toolbox_and_pop):
    tb, pop = toolbox_and_pop
    _save_hof(out_dir, pop, n=1, toolbox=tb)
    with open(out_dir / "hof_0.dill", "rb") as f:
        func = dill.load(f)
    assert callable(func)


def test_save_hof_expr_is_non_empty_string(out_dir, toolbox_and_pop):
    tb, pop = toolbox_and_pop
    _save_hof(out_dir, pop, n=1, toolbox=tb)
    expr = (out_dir / "hof_0.expr").read_text()
    assert len(expr) > 0


def test_save_hof_clamps_to_population_size(out_dir, toolbox_and_pop):
    tb, pop = toolbox_and_pop
    _save_hof(out_dir, pop, n=100, toolbox=tb)
    saved = list(out_dir.glob("hof_*.dill"))
    assert len(saved) == len(pop)
