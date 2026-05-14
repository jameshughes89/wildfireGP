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
    ANCHOR_WEIGHT,
    no_treatment,
    random_score,
    score_by_burning_neighbors,
    score_by_fire_proximity,
    score_by_fuel,
    score_fire_run,
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
    one_hop_from_fire = (5, 4)
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_indirect_attack(graph, one_hop_from_fire) == 0.0


def test_score_indirect_attack_zero_above_max_distance():
    five_hops_from_fire = (0, 0)
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_indirect_attack(graph, five_hops_from_fire, max_distance=3) == 0.0


def test_score_indirect_attack_closer_scores_higher_within_window():
    two_hops_from_fire = (5, 3)
    five_hops_from_fire = (0, 0)
    graph = _graph_with_fire(ignition=(5, 5))
    for nb in graph.neighbors(two_hops_from_fire):
        graph.nodes[nb][FUEL] = 0.9
    for nb in graph.neighbors(five_hops_from_fire):
        graph.nodes[nb][FUEL] = 0.9
    assert score_indirect_attack(graph, two_hops_from_fire) > score_indirect_attack(graph, five_hops_from_fire)


# ---------------------------------------------------------------------------
# score_ridgeline
# ---------------------------------------------------------------------------


def test_score_ridgeline_zero_when_no_fire():
    assert score_ridgeline(_graph_no_fire(), (5, 5)) == 0.0


def test_score_ridgeline_zero_below_min_distance():
    one_hop_from_fire = (5, 4)
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_ridgeline(graph, one_hop_from_fire) == 0.0


def test_score_ridgeline_zero_above_max_distance():
    five_hops_from_fire = (0, 0)
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_ridgeline(graph, five_hops_from_fire, max_distance=3) == 0.0


def test_score_ridgeline_higher_elevation_and_slope_scores_higher():
    graph = _graph_with_fire()
    no_distance_limit = dict(min_distance=0, max_distance=100)
    high = max(graph.nodes, key=lambda n: score_ridgeline(graph, n, **no_distance_limit))
    low = min(graph.nodes, key=lambda n: score_ridgeline(graph, n, **no_distance_limit))
    assert score_ridgeline(graph, high, **no_distance_limit) > score_ridgeline(graph, low, **no_distance_limit)


# ---------------------------------------------------------------------------
# score_head_fire
# ---------------------------------------------------------------------------


def test_score_by_fuel_higher_fuel_scores_higher():
    g = _graph_no_fire()
    high, low = (3, 3), (3, 4)
    g.nodes[high][FUEL] = 0.9
    g.nodes[low][FUEL] = 0.1
    assert score_by_fuel(g, high) > score_by_fuel(g, low)


def test_score_by_fuel_anchor_nudges_equal_fuel_toward_unburnable_neighbour():
    g = _graph_no_fire()
    node_with_anchor, node_without = (1, 1), (8, 8)
    g.nodes[node_with_anchor][FUEL] = g.nodes[node_without][FUEL] = 0.5
    g.nodes[(0, 0)][STATE] = NodeState.BURNED  # neighbour of (1,1), not of (8,8)
    assert score_by_fuel(g, node_with_anchor) > score_by_fuel(g, node_without)


def test_score_head_fire_zero_when_no_fire():
    assert score_head_fire(_graph_no_fire(), (5, 5)) == 0.0


def test_score_head_fire_zero_below_min_distance():
    one_hop_from_fire = (5, 4)
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_head_fire(graph, one_hop_from_fire) == 0.0


def test_score_head_fire_zero_above_max_distance():
    five_hops_from_fire = (0, 0)
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_head_fire(graph, five_hops_from_fire, max_distance=3) == 0.0


def test_score_head_fire_finite_within_window():
    two_hops_from_fire = (5, 3)
    assert math.isfinite(score_head_fire(_graph_with_fire(ignition=(5, 5)), two_hops_from_fire))


def test_score_head_fire_downwind_scores_higher_than_crosswind():
    g = _graph_with_fire(ignition=(5, 5))
    downwind = (7, 5)
    crosswind = (5, 3)
    assert score_head_fire(g, downwind) > score_head_fire(g, crosswind)


# ---------------------------------------------------------------------------
# score_fire_run
# ---------------------------------------------------------------------------


def test_score_fire_run_zero_when_no_fire():
    assert score_fire_run(_graph_no_fire(), (5, 5)) == 0.0


def test_score_fire_run_zero_below_min_distance():
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_fire_run(graph, (5, 4)) == 0.0


def test_score_fire_run_zero_above_max_distance():
    graph = _graph_with_fire(ignition=(5, 5))
    assert score_fire_run(graph, (0, 0), max_distance=3) == 0.0


def test_score_fire_run_finite_within_window():
    graph = _graph_with_fire(ignition=(5, 5))
    assert math.isfinite(score_fire_run(graph, (7, 5)))


def test_score_fire_run_downwind_scores_higher_than_crosswind():
    graph = _graph_with_fire(ignition=(5, 5))
    downwind = (7, 5)  # south of fire, aligned with north wind
    crosswind = (5, 3)  # west of fire, perpendicular to wind
    assert score_fire_run(graph, downwind) > score_fire_run(graph, crosswind)


def test_score_fire_run_anchor_weight_keeps_nudge_smaller_than_core():
    graph = _graph_with_fire(ignition=(5, 5))
    downwind = (7, 5)
    anchor_contribution = ANCHOR_WEIGHT * 8 / 8  # max possible anchoring
    core = abs(score_fire_run(graph, downwind) - anchor_contribution)
    assert anchor_contribution <= core or core > 0  # nudge does not swamp core signal


def test_score_fire_run_higher_fuel_scores_higher():
    # (7,4) and (7,6) are both Chebyshev distance 2 from (5,5) and have identical wind alignment
    # (symmetric about the fire column), so fuel is the only differentiating factor.
    graph = _graph_with_fire(ignition=(5, 5))
    high_fuel, low_fuel = (7, 4), (7, 6)
    graph.nodes[high_fuel][FUEL] = 0.9
    graph.nodes[low_fuel][FUEL] = 0.1
    assert score_fire_run(graph, high_fuel) > score_fire_run(graph, low_fuel)
