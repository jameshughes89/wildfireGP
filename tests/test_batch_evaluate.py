"""Tests for scripts/batch_evaluate.py."""

import pathlib

import dill
import numpy as np
import pytest

from scripts.batch_evaluate import _load_candidates, main
from scripts.cli import find_valid_ignition as _find_valid_ignition
from wildfireGP.network import create_grid, set_fuel_moisture, set_wind


def _const_func(graph, node):
    return 1.0


def _const_func_2(graph, node):
    return 2.0


def _write_population(path: pathlib.Path, entries: list[tuple[str, object]]) -> None:
    with open(path, "wb") as f:
        dill.dump(entries, f)


def _write_hof(directory: pathlib.Path, expr: str, func: object, index: int = 0) -> None:
    with open(directory / f"hof_{index}.dill", "wb") as f:
        dill.dump(func, f)
    (directory / f"hof_{index}.expr").write_text(expr)


# ---------------------------------------------------------------------------
# _load_candidates
# ---------------------------------------------------------------------------


def test_load_candidates_returns_population_entries(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("expr_a", _const_func), ("expr_b", _const_func_2)])
    candidates = _load_candidates(tmp_path)
    exprs = [e for e, _ in candidates]
    assert "expr_a" in exprs
    assert "expr_b" in exprs


def test_load_candidates_deduplicates_identical_expressions(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("shared_expr", _const_func)])
    _write_hof(tmp_path, "shared_expr", _const_func, index=0)
    candidates = _load_candidates(tmp_path)
    assert len(candidates) == 1


def test_load_candidates_includes_hof_only_expressions(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("expr_pop", _const_func)])
    _write_hof(tmp_path, "expr_hof", _const_func_2, index=0)
    candidates = _load_candidates(tmp_path)
    exprs = [e for e, _ in candidates]
    assert "expr_pop" in exprs
    assert "expr_hof" in exprs


def test_load_candidates_hof_without_population(tmp_path):
    _write_hof(tmp_path, "expr_hof", _const_func, index=0)
    candidates = _load_candidates(tmp_path)
    assert len(candidates) == 1


def test_load_candidates_raises_when_directory_is_empty(tmp_path):
    with pytest.raises(SystemExit):
        _load_candidates(tmp_path)


def test_load_candidates_skips_hof_dill_without_expr_file(tmp_path):
    with open(tmp_path / "hof_0.dill", "wb") as f:
        dill.dump(_const_func, f)
    with pytest.raises(SystemExit):
        _load_candidates(tmp_path)


# ---------------------------------------------------------------------------
# _find_valid_ignition
# ---------------------------------------------------------------------------


def test_find_valid_ignition_returns_a_tuple():
    g = create_grid(10, 10, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    ignition = _find_valid_ignition(
        g, np.random.default_rng(0), max_steps=50, intervention_delay=0, min_burned=0, max_tries=5
    )
    assert isinstance(ignition, tuple)


def test_find_valid_ignition_returns_node_in_graph():
    g = create_grid(10, 10, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    ignition = _find_valid_ignition(
        g, np.random.default_rng(0), max_steps=50, intervention_delay=0, min_burned=0, max_tries=5
    )
    r, c = ignition
    assert 0 <= r < g.rows and 0 <= c < g.cols


def test_find_valid_ignition_falls_back_when_min_burned_impossible(caplog):
    # min_burned set impossibly high so no ignition can satisfy it; should warn and return best
    g = create_grid(10, 10, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    import logging

    with caplog.at_level(logging.WARNING):
        ignition = _find_valid_ignition(
            g, np.random.default_rng(0), max_steps=5, intervention_delay=0, min_burned=99999, max_tries=3
        )
    assert ignition is not None
    assert "No valid ignition" in caplog.text


# ---------------------------------------------------------------------------
# main() smoke test
# ---------------------------------------------------------------------------


def test_main_runs_and_prints_table(tmp_path, capsys):
    _write_population(tmp_path / "final_population.dill", [("constant", _const_func)])
    main(
        [
            "--results-dir",
            str(tmp_path),
            "--landscapes",
            "2",
            "--runs",
            "3",
            "--seed",
            "0",
            "--rows",
            "5",
            "--cols",
            "5",
        ]
    )
    out = capsys.readouterr().out
    assert "rank" in out
    assert "mean" in out
    assert "constant" in out


def test_main_saves_csv_when_output_flag_given(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("constant", _const_func)])
    csv_path = tmp_path / "results.csv"
    main(
        [
            "--results-dir",
            str(tmp_path),
            "--landscapes",
            "2",
            "--runs",
            "3",
            "--seed",
            "0",
            "--rows",
            "5",
            "--cols",
            "5",
            "--output",
            str(csv_path),
        ]
    )
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert lines[0] == "rank,mean,std,expr"
    assert "constant" in lines[1]


def test_main_no_csv_without_output_flag(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("constant", _const_func)])
    main(
        [
            "--results-dir",
            str(tmp_path),
            "--landscapes",
            "2",
            "--runs",
            "3",
            "--seed",
            "0",
            "--rows",
            "5",
            "--cols",
            "5",
        ]
    )
    assert not any(tmp_path.glob("*.csv"))
