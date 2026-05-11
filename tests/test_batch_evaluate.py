"""Tests for scripts/batch_evaluate.py."""

import pathlib

import dill
import pytest

from scripts.batch_evaluate import _load_candidates, main


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
    _write_population(tmp_path / "final_population.dill", [("expr_a", lambda g, n: 1.0), ("expr_b", lambda g, n: 2.0)])
    candidates = _load_candidates(tmp_path)
    exprs = [e for e, _ in candidates]
    assert "expr_a" in exprs
    assert "expr_b" in exprs


def test_load_candidates_deduplicates_identical_expressions(tmp_path):
    func = lambda g, n: 1.0
    _write_population(tmp_path / "final_population.dill", [("shared_expr", func)])
    _write_hof(tmp_path, "shared_expr", func, index=0)
    candidates = _load_candidates(tmp_path)
    assert len(candidates) == 1


def test_load_candidates_includes_hof_only_expressions(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("expr_pop", lambda g, n: 1.0)])
    _write_hof(tmp_path, "expr_hof", lambda g, n: 2.0, index=0)
    candidates = _load_candidates(tmp_path)
    exprs = [e for e, _ in candidates]
    assert "expr_pop" in exprs
    assert "expr_hof" in exprs


def test_load_candidates_hof_without_population(tmp_path):
    _write_hof(tmp_path, "expr_hof", lambda g, n: 1.0, index=0)
    candidates = _load_candidates(tmp_path)
    assert len(candidates) == 1


def test_load_candidates_raises_when_directory_is_empty(tmp_path):
    with pytest.raises(SystemExit):
        _load_candidates(tmp_path)


def test_load_candidates_skips_hof_dill_without_expr_file(tmp_path):
    with open(tmp_path / "hof_0.dill", "wb") as f:
        dill.dump(lambda g, n: 1.0, f)
    with pytest.raises(SystemExit):
        _load_candidates(tmp_path)


# ---------------------------------------------------------------------------
# main() smoke test
# ---------------------------------------------------------------------------


def test_main_runs_and_prints_table(tmp_path, capsys):
    _write_population(tmp_path / "final_population.dill", [("constant", lambda g, n: 1.0)])
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
