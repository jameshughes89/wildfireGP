import math

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


# ---------------------------------------------------------------------------
# score_indirect_attack
# ---------------------------------------------------------------------------


def test_score_indirect_attack_zero_when_no_fire():
    assert score_indirect_attack(_graph_no_fire(), (5, 5)) == 0.0


def test_score_indirect_attack_zero_below_min_distance():
    # (5, 4) is 1 hop from fire at (5, 5) — below default min_distance=2
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_indirect_attack(graph, (5, 4)) == 0.0


def test_score_indirect_attack_zero_above_max_distance():
    # (0, 0) is 5 hops from fire at (5, 5); force max_distance=3 to put it outside
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_indirect_attack(graph, (0, 0), max_distance=3) == 0.0


def test_score_indirect_attack_closer_scores_higher_within_window():
    # (5, 3) is 2 hops from fire, (0, 0) is 5 hops — both within default window
    graph = _graph_with_fire(ignition=(5, 5))
    for nb in graph.neighbors((5, 3)):
        graph.nodes[nb][FUEL] = 0.9
    for nb in graph.neighbors((0, 0)):
        graph.nodes[nb][FUEL] = 0.9
    assert score_indirect_attack(graph, (5, 3)) > score_indirect_attack(graph, (0, 0))


# ---------------------------------------------------------------------------
# score_ridgeline
# ---------------------------------------------------------------------------


def test_score_ridgeline_zero_when_no_fire():
    assert score_ridgeline(_graph_no_fire(), (5, 5)) == 0.0


def test_score_ridgeline_zero_below_min_distance():
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_ridgeline(graph, (5, 4)) == 0.0


def test_score_ridgeline_zero_above_max_distance():
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_ridgeline(graph, (0, 0), max_distance=3) == 0.0


def test_score_ridgeline_higher_elevation_and_slope_scores_higher():
    # Disable distance window so the test focuses purely on topographic ranking
    graph = _graph_with_fire()
    high = max(graph.nodes, key=lambda n: score_ridgeline(graph, n, min_distance=0, max_distance=100))
    low = min(graph.nodes, key=lambda n: score_ridgeline(graph, n, min_distance=0, max_distance=100))
    assert score_ridgeline(graph, high, min_distance=0, max_distance=100) > score_ridgeline(
        graph, low, min_distance=0, max_distance=100
    )


# ---------------------------------------------------------------------------
# score_head_fire
# ---------------------------------------------------------------------------


def test_score_head_fire_zero_when_no_fire():
    assert score_head_fire(_graph_no_fire(), (5, 5)) == 0.0


def test_score_head_fire_zero_below_min_distance():
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_head_fire(graph, (5, 4)) == 0.0


def test_score_head_fire_zero_above_max_distance():
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_head_fire(graph, (0, 0), max_distance=3) == 0.0


def test_score_head_fire_finite_within_window():
    # (5, 3) is 2 hops from fire at (5, 5) — within default window
    assert math.isfinite(score_head_fire(_graph_with_fire(ignition=(5, 5)), (5, 3)))
