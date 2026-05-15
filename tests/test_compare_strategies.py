import dill
import numpy as np
import pytest

from scripts.compare_strategies import _load_strategies, _run_strategy, main
from wildfireGP.network import (
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.strategies import ALL_STRATEGIES, random_score


def _write_population(path, entries):
    with open(path, "wb") as f:
        dill.dump(entries, f)


def _make_landscapes(n: int = 1, rows: int = 10, cols: int = 10) -> tuple[list, list[list]]:
    """Return (landscapes, run_seed_matrix) for n landscapes on a rows×cols grid."""
    landscapes = []
    run_seed_matrix = []
    for i in range(n):
        seed = i
        g = create_grid(rows, cols, seed=seed)
        set_wind(g, speed=20.0, direction=0.0)
        set_fuel_moisture(g, moisture=0.2)
        ignition = select_ignition_node(g, np.random.default_rng(seed))
        landscapes.append((seed, ignition))
        run_seed_matrix.append([seed * 100 + j for j in range(5)])
    return landscapes, run_seed_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dill_file(tmp_path):
    path = tmp_path / "hof_0.dill"
    with open(path, "wb") as f:
        dill.dump(random_score, f)
    return path


# ---------------------------------------------------------------------------
# _run_strategy
# ---------------------------------------------------------------------------


def test_run_strategy_returns_arrays_of_correct_length():
    landscapes, run_seed_matrix = _make_landscapes(n=2)
    burned, peak = _run_strategy(random_score, landscapes, 2, 10, 10, 20.0, 0.0, 0.2, 20, run_seed_matrix)
    assert len(burned) == 10  # 2 landscapes × 5 runs
    assert len(peak) == 10


def test_run_strategy_burned_values_are_non_negative():
    landscapes, run_seed_matrix = _make_landscapes()
    burned, _ = _run_strategy(random_score, landscapes, 2, 10, 10, 20.0, 0.0, 0.2, 20, run_seed_matrix)
    assert all(b >= 0 for b in burned)


def test_run_strategy_peak_values_are_non_negative():
    landscapes, run_seed_matrix = _make_landscapes()
    _, peak = _run_strategy(random_score, landscapes, 2, 10, 10, 20.0, 0.0, 0.2, 20, run_seed_matrix)
    assert all(p >= 0 for p in peak)


def test_run_strategy_different_seeds_produce_variation():
    landscapes, run_seed_matrix = _make_landscapes(n=2)
    # Use 10 distinct run seeds across 2 landscapes
    run_seed_matrix = [[i * 7 + j for j in range(5)] for i in range(2)]
    burned, _ = _run_strategy(random_score, landscapes, 0, 10, 10, 20.0, 0.0, 0.2, 20, run_seed_matrix)
    assert burned.std() > 0


# ---------------------------------------------------------------------------
# _load_strategies
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, hof=None, results_dir=None, expr=None):
        self.hof = hof
        self.results_dir = results_dir
        self.expr = expr


def test_load_strategies_baselines_always_present():
    strategies = _load_strategies(_FakeArgs())
    names = [n for n, _ in strategies]
    assert all(f.__name__ in names for f in ALL_STRATEGIES)


def test_main_includes_no_treatment_baseline(capsys):
    main(
        [
            "--rows",
            "10",
            "--cols",
            "10",
            "--seed",
            "0",
            "--runs",
            "2",
            "--landscapes",
            "1",
            "--max-steps",
            "10",
            "--min-burned",
            "0",
        ]
    )
    output = capsys.readouterr().out
    assert "no_treatment" in output


def test_load_strategies_dill_file_loaded(dill_file):
    strategies = _load_strategies(_FakeArgs(hof=[dill_file]))
    names = [n for n, _ in strategies]
    assert "hof_0" in names


def test_load_strategies_results_dir_loads_dill_files(tmp_path, dill_file):
    strategies = _load_strategies(_FakeArgs(results_dir=tmp_path))
    names = [n for n, _ in strategies]
    assert "hof_0" in names


def test_load_strategies_results_dir_does_not_load_final_population_dill(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("expr_a", random_score)])
    strategies = _load_strategies(_FakeArgs(results_dir=tmp_path))
    names = [n for n, _ in strategies]
    assert "final_population" not in names


def test_load_strategies_expr_loads_candidate(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("my_expr", random_score)])
    strategies = _load_strategies(_FakeArgs(results_dir=tmp_path, expr="my_expr"))
    names = [n for n, _ in strategies]
    assert any("my_expr" in n for n in names)


def test_load_strategies_expr_without_results_dir_raises():
    with pytest.raises(SystemExit):
        _load_strategies(_FakeArgs(expr="my_expr"))


def test_load_strategies_deduplicates_dill_paths(dill_file):
    strategies = _load_strategies(_FakeArgs(hof=[dill_file, dill_file]))
    names = [n for n, _ in strategies]
    assert names.count("hof_0") == 1


def test_load_strategies_dill_callable_is_callable(dill_file):
    strategies = _load_strategies(_FakeArgs(hof=[dill_file]))
    func = next(f for n, f in strategies if n == "hof_0")
    assert callable(func)
