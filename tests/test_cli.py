"""Tests for scripts/cli.py shared utilities."""

import pytest

from scripts.cli import load_candidate_by_expr, load_expr

_EXPR = "fuel_level(graph, node)"
_OTHER_EXPR = "elevation(graph, node)"


def _write_population(directory, exprs):
    (directory / "final_population.expr").write_text("\n".join(exprs) + "\n")


def _write_hof(directory, expr, index=0):
    (directory / f"hof_{index}.expr").write_text(expr)


def test_load_expr_returns_callable():
    func = load_expr(_EXPR)
    assert callable(func)


def test_load_expr_unknown_primitive_raises():
    with pytest.raises(Exception):
        load_expr("nonexistent_primitive(graph, node)")


def test_load_candidate_by_expr_finds_in_population(tmp_path):
    _write_population(tmp_path, [_EXPR])
    assert callable(load_candidate_by_expr(tmp_path, _EXPR))


def test_load_candidate_by_expr_falls_back_to_hof(tmp_path):
    _write_hof(tmp_path, _EXPR, index=0)
    assert callable(load_candidate_by_expr(tmp_path, _EXPR))


def test_load_candidate_by_expr_raises_when_not_found(tmp_path):
    _write_population(tmp_path, [_OTHER_EXPR])
    with pytest.raises(SystemExit):
        load_candidate_by_expr(tmp_path, _EXPR)


def test_load_candidate_by_expr_raises_when_no_sources(tmp_path):
    with pytest.raises(SystemExit):
        load_candidate_by_expr(tmp_path, _EXPR)
