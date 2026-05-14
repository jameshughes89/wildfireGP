"""Tests for scripts/cli.py shared utilities."""

import dill
import pytest

from scripts.cli import load_candidate_by_expr
from wildfireGP.strategies import random_score, score_by_fire_proximity


def _write_population(path, entries):
    with open(path, "wb") as f:
        dill.dump(entries, f)


def _write_hof(directory, expr, func, index=0):
    with open(directory / f"hof_{index}.dill", "wb") as f:
        dill.dump(func, f)
    (directory / f"hof_{index}.expr").write_text(expr)


def test_load_candidate_by_expr_finds_in_population(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("my_expr", random_score)])
    func = load_candidate_by_expr(tmp_path, "my_expr")
    assert callable(func)


def test_load_candidate_by_expr_falls_back_to_hof(tmp_path):
    _write_hof(tmp_path, "hof_expr", random_score, index=0)
    func = load_candidate_by_expr(tmp_path, "hof_expr")
    assert callable(func)


def test_load_candidate_by_expr_population_takes_precedence(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("shared", random_score)])
    _write_hof(tmp_path, "shared", score_by_fire_proximity, index=0)
    func = load_candidate_by_expr(tmp_path, "shared")
    assert func is random_score


def test_load_candidate_by_expr_raises_when_not_found(tmp_path):
    _write_population(tmp_path / "final_population.dill", [("other_expr", random_score)])
    with pytest.raises(SystemExit):
        load_candidate_by_expr(tmp_path, "nonexistent")


def test_load_candidate_by_expr_raises_when_no_sources(tmp_path):
    with pytest.raises(SystemExit):
        load_candidate_by_expr(tmp_path, "any_expr")


def test_load_candidate_by_expr_hof_without_dill_skipped(tmp_path):
    (tmp_path / "hof_0.expr").write_text("my_expr")
    with pytest.raises(SystemExit):
        load_candidate_by_expr(tmp_path, "my_expr")
