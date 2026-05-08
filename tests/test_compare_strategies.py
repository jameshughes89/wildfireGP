import dill
import numpy as np
import pytest

from scripts.compare_strategies import _load_strategies, _print_table, _run_strategy
from wildfireGP.network import (
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.strategies import ALL_STRATEGIES, no_treatment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph():
    g = create_grid(10, 10, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    return g


@pytest.fixture()
def ignition(graph):
    return select_ignition_node(graph, np.random.default_rng(0))


@pytest.fixture()
def dill_file(tmp_path):
    path = tmp_path / "hof_0.dill"
    with open(path, "wb") as f:
        dill.dump(no_treatment, f)
    return path


# ---------------------------------------------------------------------------
# _run_strategy
# ---------------------------------------------------------------------------


def test_run_strategy_returns_arrays_of_correct_length(graph, ignition):
    burned, peak = _run_strategy(no_treatment, graph, [ignition], 2, 20, runs=5, base_seed=0)
    assert len(burned) == 5
    assert len(peak) == 5


def test_run_strategy_burned_values_are_non_negative(graph, ignition):
    burned, _ = _run_strategy(no_treatment, graph, [ignition], 2, 20, runs=5, base_seed=0)
    assert all(b >= 0 for b in burned)


def test_run_strategy_peak_values_are_non_negative(graph, ignition):
    _, peak = _run_strategy(no_treatment, graph, [ignition], 2, 20, runs=5, base_seed=0)
    assert all(p >= 0 for p in peak)


def test_run_strategy_different_seeds_produce_variation(graph, ignition):
    burned, _ = _run_strategy(no_treatment, graph, [ignition], 0, 20, runs=10, base_seed=0)
    assert burned.std() > 0


# ---------------------------------------------------------------------------
# _load_strategies
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, hof=None, results_dir=None):
        self.hof = hof
        self.results_dir = results_dir


def test_load_strategies_baselines_always_present():
    strategies = _load_strategies(_FakeArgs())
    names = [n for n, _ in strategies]
    assert all(f.__name__ in names for f in ALL_STRATEGIES)


def test_load_strategies_dill_file_loaded(dill_file):
    strategies = _load_strategies(_FakeArgs(hof=[dill_file]))
    names = [n for n, _ in strategies]
    assert "hof_0" in names


def test_load_strategies_results_dir_loads_dill_files(tmp_path, dill_file):
    strategies = _load_strategies(_FakeArgs(results_dir=tmp_path))
    names = [n for n, _ in strategies]
    assert "hof_0" in names


def test_load_strategies_deduplicates_dill_paths(dill_file):
    strategies = _load_strategies(_FakeArgs(hof=[dill_file, dill_file]))
    names = [n for n, _ in strategies]
    assert names.count("hof_0") == 1


def test_load_strategies_dill_callable_is_callable(dill_file):
    strategies = _load_strategies(_FakeArgs(hof=[dill_file]))
    func = next(f for n, f in strategies if n == "hof_0")
    assert callable(func)


# ---------------------------------------------------------------------------
# _print_table
# ---------------------------------------------------------------------------


def test_print_table_runs_without_error(capsys):
    results = [("no_treatment", 50.0, 5.0, 10.0, 2.0), ("random_score", 40.0, 4.0, 8.0, 1.5)]
    _print_table(results)
    out = capsys.readouterr().out
    assert "no_treatment" in out
    assert "random_score" in out


def test_print_table_contains_numeric_values(capsys):
    _print_table([("no_treatment", 50.12, 5.34, 10.56, 2.78)])
    out = capsys.readouterr().out
    assert "50.12" in out
    assert "10.56" in out
