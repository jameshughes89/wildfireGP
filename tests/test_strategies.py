import math

import pytest

from wildfireGP.features import precompute_burnable_fire_map, precompute_fire_map
from wildfireGP.network import (
    BURN_TIMER,
    FUEL,
    STATE,
    NodeState,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.spread import MAX_BURN_STEPS
from wildfireGP.strategies import (
    no_treatment,
    random_score,
    score_by_burning_neighbors,
    score_by_fire_proximity,
    score_head_fire,
    score_indirect_attack,
    score_ridgeline,
)


def _graph_no_fire():
    g = create_grid(10, 10, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    precompute_fire_map(g)
    precompute_burnable_fire_map(g)
    return g


def _graph_with_fire(ignition=(5, 5)):
    g = create_grid(10, 10, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    g.nodes[ignition][STATE] = NodeState.BURNING
    g.nodes[ignition][BURN_TIMER] = max(1, math.ceil(g.nodes[ignition][FUEL] * MAX_BURN_STEPS))
    precompute_fire_map(g)
    precompute_burnable_fire_map(g)
    return g


def test_no_treatment_always_zero():
    graph = _graph_with_fire()
    assert all(no_treatment(graph, n) == 0.0 for n in graph.nodes)


def test_random_score_varies_across_nodes():
    graph = _graph_with_fire()
    scores = {random_score(graph, n) for n in graph.nodes}
    assert len(scores) > 1


def test_score_by_fire_proximity_closer_scores_higher():
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_by_fire_proximity(graph, (5, 4)) > score_by_fire_proximity(graph, (0, 0))


def test_score_by_burning_neighbors_more_neighbors_scores_higher():
    graph = _graph_no_fire()
    node = (5, 5)
    for nb in list(graph.neighbors(node))[:3]:
        graph.nodes[nb][STATE] = NodeState.BURNING
    precompute_fire_map(graph)
    assert score_by_burning_neighbors(graph, node) > score_by_burning_neighbors(graph, (0, 0))


def test_score_indirect_attack_zero_when_no_fire():
    assert score_indirect_attack(_graph_no_fire(), (5, 5)) == 0.0


def test_score_indirect_attack_closer_to_fire_scores_higher():
    graph = _graph_with_fire(ignition=(5, 5))
    for nb in graph.neighbors((5, 4)):
        graph.nodes[nb][FUEL] = 0.9
    for nb in graph.neighbors((0, 0)):
        graph.nodes[nb][FUEL] = 0.9
    assert score_indirect_attack(graph, (5, 4)) > score_indirect_attack(graph, (0, 0))


def test_score_ridgeline_higher_elevation_and_slope_scores_higher():
    graph = _graph_with_fire()
    high = max(graph.nodes, key=lambda n: score_ridgeline(graph, n))
    low = min(graph.nodes, key=lambda n: score_ridgeline(graph, n))
    assert score_ridgeline(graph, high) > score_ridgeline(graph, low)


def test_score_head_fire_zero_when_no_fire():
    assert score_head_fire(_graph_no_fire(), (5, 5)) == 0.0


def test_score_head_fire_finite_with_fire():
    assert math.isfinite(score_head_fire(_graph_with_fire(ignition=(5, 5)), (5, 3)))
