import math

import pytest

from wildfireGP.features import precompute_burnable_fire_map, precompute_fire_map
from wildfireGP.network import (
    BURN_TIMER,
    ELEVATION,
    FUEL,
    SLOPE,
    STATE,
    NodeState,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.spread import MAX_BURN_STEPS
from wildfireGP.strategies import (
    ALL_STRATEGIES,
    no_treatment,
    random_score,
    score_by_burning_neighbors,
    score_by_fire_proximity,
    score_by_fuel,
    score_head_fire,
    score_indirect_attack,
    score_ridgeline,
)


def _graph_no_fire():
    g = create_grid(10, 10, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    return g


def _graph_with_fire(ignition=(5, 5)):
    g = _graph_no_fire()
    g.nodes[ignition][STATE] = NodeState.BURNING
    g.nodes[ignition][BURN_TIMER] = max(1, math.ceil(g.nodes[ignition][FUEL] * MAX_BURN_STEPS))
    precompute_fire_map(g)
    precompute_burnable_fire_map(g)
    return g


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_strategy_returns_float(strategy):
    graph = _graph_with_fire()
    node = (3, 3)
    result = strategy(graph, node)
    assert isinstance(result, (int, float))


def test_no_treatment_always_zero():
    graph = _graph_with_fire()
    for node in list(graph.nodes)[:10]:
        assert no_treatment(graph, node) == 0.0


def test_random_score_varies_across_nodes():
    graph = _graph_with_fire()
    nodes = list(graph.nodes)
    scores = {random_score(graph, n) for n in nodes}
    assert len(scores) > 1


def test_score_by_fuel_matches_fuel_attribute():
    graph = _graph_with_fire()
    node = (3, 3)
    assert score_by_fuel(graph, node) == graph.nodes[node][FUEL]


def test_score_by_fire_proximity_closer_scores_higher():
    graph = _graph_with_fire(ignition=(5, 5))
    close = (5, 4)
    far = (0, 0)
    assert score_by_fire_proximity(graph, close) > score_by_fire_proximity(graph, far)


def test_score_by_burning_neighbors_more_neighbors_scores_higher():
    graph = _graph_no_fire()
    node = (5, 5)
    for nb in list(graph.neighbors(node))[:3]:
        graph.nodes[nb][STATE] = NodeState.BURNING
    precompute_fire_map(graph)
    precompute_burnable_fire_map(graph)
    isolated = (0, 0)
    assert score_by_burning_neighbors(graph, node) > score_by_burning_neighbors(graph, isolated)


def test_score_indirect_attack_zero_when_no_fire():
    graph = _graph_no_fire()
    precompute_fire_map(graph)
    precompute_burnable_fire_map(graph)
    assert score_indirect_attack(graph, (5, 5)) == 0.0


def test_score_indirect_attack_closer_to_fire_scores_higher():
    graph = _graph_with_fire(ignition=(5, 5))
    graph.nodes[(5, 4)][FUEL] = 0.9
    graph.nodes[(0, 0)][FUEL] = 0.9
    for nb in graph.neighbors((5, 4)):
        graph.nodes[nb][FUEL] = 0.9
    for nb in graph.neighbors((0, 0)):
        graph.nodes[nb][FUEL] = 0.9
    close = (5, 4)
    far = (0, 0)
    assert score_indirect_attack(graph, close) > score_indirect_attack(graph, far)


def test_score_ridgeline_matches_elevation_plus_slope():
    graph = _graph_with_fire()
    node = (3, 3)
    expected = graph.nodes[node][ELEVATION] + graph.nodes[node][SLOPE]
    assert score_ridgeline(graph, node) == pytest.approx(expected)


def test_score_ridgeline_higher_elevation_and_slope_scores_higher():
    graph = _graph_with_fire()
    high = max(graph.nodes, key=lambda n: graph.nodes[n][ELEVATION] + graph.nodes[n][SLOPE])
    low = min(graph.nodes, key=lambda n: graph.nodes[n][ELEVATION] + graph.nodes[n][SLOPE])
    assert score_ridgeline(graph, high) > score_ridgeline(graph, low)


def test_score_head_fire_zero_when_no_fire():
    graph = _graph_no_fire()
    precompute_fire_map(graph)
    precompute_burnable_fire_map(graph)
    assert score_head_fire(graph, (5, 5)) == 0.0


def test_score_head_fire_returns_finite_value_with_fire():
    graph = _graph_with_fire(ignition=(5, 5))
    score = score_head_fire(graph, (5, 3))
    assert math.isfinite(score)
