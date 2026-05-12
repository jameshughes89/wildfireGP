"""Tests for scripts/batch_evaluate.py."""

import pathlib

import dill
import pytest

from scripts.batch_evaluate import _load_candidates, main


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
# main() smoke test
# ---------------------------------------------------------------------------


def test_main_runs_and_prints_table(tmp_path, capsys):
    _write_population(tmp_path / "final_population.dill", [("constant", _const_func)])
    main(
        [
            "--results-dir",
            str(tmp_path),
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
