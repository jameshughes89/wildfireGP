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

_EXPR = "fuel_level(graph, node)"


def _make_landscapes(n: int = 1, rows: int = 10, cols: int = 10) -> tuple[list, list[list]]:
    landscapes = []
    run_seed_matrix = []
    for i in range(n):
        seed = i
        g = create_grid(rows, cols, seed=seed)
        set_wind(g, speed=20.0, direction=0.0)
        set_fuel_moisture(g, moisture=0.2)
        ignition = select_ignition_node(g, np.random.default_rng(seed))
        landscapes.append((seed, 20.0, 0.0, ignition))
        run_seed_matrix.append([seed * 100 + j for j in range(5)])
    return landscapes, run_seed_matrix


@pytest.fixture()
def hof_file(tmp_path):
    path = tmp_path / "hof_0.expr"
    path.write_text(_EXPR)
    return path


# ---------------------------------------------------------------------------
# _run_strategy
# ---------------------------------------------------------------------------


def test_run_strategy_returns_arrays_of_correct_length():
    landscapes, run_seed_matrix = _make_landscapes(n=2)
    burned, peak = _run_strategy(random_score, landscapes, 2, {"rows": 10, "cols": 10}, 0.2, 20, run_seed_matrix)
    assert len(burned) == 10
    assert len(peak) == 10


def test_run_strategy_burned_values_are_non_negative():
    landscapes, run_seed_matrix = _make_landscapes()
    burned, _ = _run_strategy(random_score, landscapes, 2, {"rows": 10, "cols": 10}, 0.2, 20, run_seed_matrix)
    assert all(b >= 0 for b in burned)


def test_run_strategy_peak_values_are_non_negative():
    landscapes, run_seed_matrix = _make_landscapes()
    _, peak = _run_strategy(random_score, landscapes, 2, {"rows": 10, "cols": 10}, 0.2, 20, run_seed_matrix)
    assert all(p >= 0 for p in peak)


def test_run_strategy_different_seeds_produce_variation():
    landscapes, run_seed_matrix = _make_landscapes(n=2)
    run_seed_matrix = [[i * 7 + j for j in range(5)] for i in range(2)]
    burned, _ = _run_strategy(random_score, landscapes, 0, {"rows": 10, "cols": 10}, 0.2, 20, run_seed_matrix)
    assert burned.std() > 0


# ---------------------------------------------------------------------------
# _load_strategies
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, hof=None, results_dir=None, expr=None, strategies=None, no_baselines=False):
        self.hof = hof
        self.results_dir = results_dir
        self.expr = expr
        self.strategies = strategies
        self.no_baselines = no_baselines


def test_load_strategies_baselines_always_present():
    strategies = _load_strategies(_FakeArgs())
    names = [n for n, _ in strategies]
    assert all(f.__name__ in names for f in ALL_STRATEGIES)


def test_load_strategies_filter_keeps_only_requested():
    strategies = _load_strategies(_FakeArgs(strategies="score_by_fire_proximity,random_score"))
    names = {n for n, _ in strategies}
    assert names == {"score_by_fire_proximity", "random_score"}


def test_load_strategies_filter_tolerates_whitespace():
    strategies = _load_strategies(_FakeArgs(strategies=" score_by_fire_proximity , random_score "))
    names = {n for n, _ in strategies}
    assert names == {"score_by_fire_proximity", "random_score"}


def test_load_strategies_filter_rejects_unknown():
    with pytest.raises(SystemExit) as exc_info:
        _load_strategies(_FakeArgs(strategies="score_by_fire_proximity,nonexistent_strategy"))
    assert "nonexistent_strategy" in str(exc_info.value)


def test_load_strategies_filter_accepts_no_treatment():
    # no_treatment is not in ALL_STRATEGIES (handled specially in main), but the filter should accept it as a
    # valid name so users can request it via --strategies.
    strategies = _load_strategies(_FakeArgs(strategies="score_by_fire_proximity,no_treatment"))
    names = {n for n, _ in strategies}
    # no_treatment doesn't appear in the returned list (it's added by main); the filter just allows the name.
    assert names == {"score_by_fire_proximity"}


def test_load_strategies_filter_combined_with_hof(hof_file):
    strategies = _load_strategies(_FakeArgs(strategies="score_by_fire_proximity", hof=[hof_file]))
    names = {n for n, _ in strategies}
    assert "score_by_fire_proximity" in names
    assert hof_file.stem in names


def test_save_raw_csv_is_non_empty(tmp_path):
    raw_path = tmp_path / "raw.csv"
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
            "--save-raw",
            str(raw_path),
        ]
    )
    import csv as _csv

    with open(raw_path) as f:
        rdr = _csv.reader(f)
        next(rdr)  # header
        rows = list(rdr)
    assert len(rows) > 0


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


def test_load_strategies_hof_file_loaded(hof_file):
    strategies = _load_strategies(_FakeArgs(hof=[hof_file]))
    names = [n for n, _ in strategies]
    assert "hof_0" in names


def test_load_strategies_results_dir_loads_hof_files(tmp_path, hof_file):
    strategies = _load_strategies(_FakeArgs(results_dir=tmp_path))
    names = [n for n, _ in strategies]
    assert "hof_0" in names


def test_load_strategies_expr_loads_candidate(tmp_path):
    (tmp_path / "final_population.expr").write_text(_EXPR + "\n")
    strategies = _load_strategies(_FakeArgs(results_dir=tmp_path, expr=_EXPR))
    names = [n for n, _ in strategies]
    assert any(_EXPR in n for n in names)


def test_load_strategies_expr_without_results_dir_raises():
    with pytest.raises(SystemExit):
        _load_strategies(_FakeArgs(expr=_EXPR))


def test_load_strategies_deduplicates_hof_paths(hof_file):
    strategies = _load_strategies(_FakeArgs(hof=[hof_file, hof_file]))
    names = [n for n, _ in strategies]
    assert names.count("hof_0") == 1


def test_load_strategies_hof_callable_is_callable(hof_file):
    strategies = _load_strategies(_FakeArgs(hof=[hof_file]))
    func = next(f for n, f in strategies if n == "hof_0")
    assert callable(func)
