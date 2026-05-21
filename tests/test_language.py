import pytest
from deap import gp

from wildfireGP.features import fuel_level
from wildfireGP.language import (
    PRIMITIVE_SET,
    _if_positive,
    _max2,
    _min2,
    _protected_div,
)
from wildfireGP.network import create_grid, set_fuel_moisture, set_wind


def _graph():
    g = create_grid(3, 3, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    return g


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def test_protected_div_nonzero_returns_quotient():
    assert _protected_div(6.0, 2.0) == 3.0


def test_protected_div_zero_denominator_returns_one():
    assert _protected_div(5.0, 0.0) == 1.0


def test_if_positive_positive_condition_returns_first():
    assert _if_positive(1.0, 10.0, 20.0) == 10.0


def test_if_positive_negative_condition_returns_second():
    assert _if_positive(-1.0, 10.0, 20.0) == 20.0


def test_if_positive_zero_condition_returns_second():
    assert _if_positive(0.0, 10.0, 20.0) == 20.0


def test_max2_returns_larger():
    assert _max2(3.0, 7.0) == 7.0


def test_min2_returns_smaller():
    assert _min2(3.0, 7.0) == 3.0


# ---------------------------------------------------------------------------
# Compiled tree evaluation
# ---------------------------------------------------------------------------


def test_compiled_node_feature_tree_evaluates():
    g = _graph()
    node = (1, 1)
    tree = gp.PrimitiveTree.from_string("fuel_level(graph, node)", PRIMITIVE_SET)
    func = gp.compile(tree, PRIMITIVE_SET)
    assert func(g, node) == pytest.approx(fuel_level(g, node))


def test_compiled_graph_feature_tree_evaluates():
    g = _graph()
    node = (1, 1)
    tree = gp.PrimitiveTree.from_string("wind_speed(graph)", PRIMITIVE_SET)
    func = gp.compile(tree, PRIMITIVE_SET)
    assert func(g, node) == pytest.approx(10.0)


def test_compiled_arithmetic_tree_evaluates():
    g = _graph()
    node = (1, 1)
    tree = gp.PrimitiveTree.from_string("add(fuel_level(graph, node), fuel_level(graph, node))", PRIMITIVE_SET)
    func = gp.compile(tree, PRIMITIVE_SET)
    assert func(g, node) == pytest.approx(2.0 * fuel_level(g, node))


def test_compiled_tree_score_differs_across_nodes():
    g = _graph()
    tree = gp.PrimitiveTree.from_string("fuel_level(graph, node)", PRIMITIVE_SET)
    func = gp.compile(tree, PRIMITIVE_SET)
    scores = {node: func(g, node) for node in g.nodes()}
    assert len(set(scores.values())) > 1


def test_compiled_protected_div_tree_handles_zero():
    g = _graph()
    node = (1, 1)
    tree = gp.PrimitiveTree.from_string("div(fuel_level(graph, node), fuel_level(graph, node))", PRIMITIVE_SET)
    func = gp.compile(tree, PRIMITIVE_SET)
    result = func(g, node)
    assert result == pytest.approx(1.0)
