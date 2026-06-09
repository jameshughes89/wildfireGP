import math

from wildfireGP.features import (
    precompute_betweenness_dynamic,
    precompute_betweenness_static,
    precompute_burnable_fire_map,
    precompute_fire_map,
    precompute_mtt_pathway_count,
    precompute_neighbourhood_maps,
    precompute_reachable_unburned_area,
)
from wildfireGP.network import (
    NodeState,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.spread import MAX_BURN_STEPS
from wildfireGP.strategies import (
    random_score,
    score_anchor_flank,
    score_betweenness_dynamic,
    score_betweenness_dynamic_anchored,
    score_betweenness_static,
    score_betweenness_static_anchored,
    score_bottleneck,
    score_by_burning_neighbors,
    score_by_fire_proximity,
    score_by_fuel,
    score_composite_threat,
    score_fire_run,
    score_flank_attack,
    score_frontier_protect,
    score_frontier_protect_anchored,
    score_head_fire,
    score_indirect_attack,
    score_mtt_pathway,
    score_mtt_pathway_anchored,
    score_ridgeline,
    score_uphill_interception,
)


def _state_no_fire():
    s = create_grid(10, 10, seed=0)
    set_wind(s, speed=20.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.2)
    precompute_fire_map(s)
    precompute_burnable_fire_map(s)
    return s


def _state_with_fire(ignition=(5, 5)):
    s = create_grid(10, 10, seed=0)
    set_wind(s, speed=20.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.2)
    s.state[ignition] = NodeState.BURNING
    s.burn_timer[ignition] = max(1, math.ceil(float(s.fuel[ignition]) * MAX_BURN_STEPS))
    precompute_fire_map(s)
    precompute_burnable_fire_map(s)
    return s


def test_random_score_varies_across_cells():
    s = _state_with_fire()
    scores = {random_score(s, n) for n in s.nodes()}
    assert len(scores) > 1


def test_score_by_fire_proximity_closer_scores_higher():
    s = _state_with_fire(ignition=(5, 5))
    assert score_by_fire_proximity(s, (5, 4)) > score_by_fire_proximity(s, (0, 0))


def test_score_by_burning_neighbors_more_neighbors_scores_higher():
    s = _state_no_fire()
    node = (5, 5)
    for nb in s.neighbours(node)[:3]:
        s.state[nb] = NodeState.BURNING
    precompute_fire_map(s)
    assert score_by_burning_neighbors(s, node) > score_by_burning_neighbors(s, (0, 0))


def test_score_indirect_attack_zero_when_no_fire():
    assert score_indirect_attack(_state_no_fire(), (5, 5)) == 0.0


def test_score_indirect_attack_zero_below_min_distance():
    one_hop_from_fire = (5, 4)
    s = _state_with_fire(ignition=(5, 5))
    assert score_indirect_attack(s, one_hop_from_fire) == 0.0


def test_score_indirect_attack_zero_above_max_distance():
    five_hops_from_fire = (0, 0)
    s = _state_with_fire(ignition=(5, 5))
    assert score_indirect_attack(s, five_hops_from_fire, max_distance=3) == 0.0


def test_score_indirect_attack_closer_scores_higher_within_window():
    two_hops_from_fire = (5, 3)
    five_hops_from_fire = (0, 0)
    s = _state_with_fire(ignition=(5, 5))
    for nb in s.neighbours(two_hops_from_fire):
        s.fuel[nb] = 0.9
    for nb in s.neighbours(five_hops_from_fire):
        s.fuel[nb] = 0.9
    assert score_indirect_attack(s, two_hops_from_fire) > score_indirect_attack(s, five_hops_from_fire)


def test_score_ridgeline_zero_when_no_fire():
    assert score_ridgeline(_state_no_fire(), (5, 5)) == 0.0


def test_score_ridgeline_zero_below_min_distance():
    one_hop_from_fire = (5, 4)
    s = _state_with_fire(ignition=(5, 5))
    assert score_ridgeline(s, one_hop_from_fire) == 0.0


def test_score_ridgeline_zero_above_max_distance():
    five_hops_from_fire = (0, 0)
    s = _state_with_fire(ignition=(5, 5))
    assert score_ridgeline(s, five_hops_from_fire, max_distance=3) == 0.0


def test_score_ridgeline_higher_elevation_and_slope_scores_higher():
    s = _state_with_fire()
    no_distance_limit = dict(min_distance=0, max_distance=100)
    nodes = list(s.nodes())
    high = max(nodes, key=lambda n: score_ridgeline(s, n, **no_distance_limit))
    low = min(nodes, key=lambda n: score_ridgeline(s, n, **no_distance_limit))
    assert score_ridgeline(s, high, **no_distance_limit) > score_ridgeline(s, low, **no_distance_limit)


def test_score_by_fuel_higher_fuel_scores_higher():
    s = _state_no_fire()
    high, low = (3, 3), (3, 4)
    s.fuel[high] = 0.9
    s.fuel[low] = 0.1
    assert score_by_fuel(s, high) > score_by_fuel(s, low)


def test_score_by_fuel_anchor_nudges_equal_fuel_toward_cell_with_treated_neighbour():
    s = _state_no_fire()
    node_with_anchor, node_without = (1, 1), (8, 8)
    s.fuel[node_with_anchor] = s.fuel[node_without] = 0.5
    s.state[0, 0] = NodeState.TREATED
    assert score_by_fuel(s, node_with_anchor) > score_by_fuel(s, node_without)


def test_score_head_fire_zero_when_no_fire():
    assert score_head_fire(_state_no_fire(), (5, 5)) == 0.0


def test_score_head_fire_zero_below_min_distance():
    one_hop_from_fire = (5, 4)
    s = _state_with_fire(ignition=(5, 5))
    assert score_head_fire(s, one_hop_from_fire) == 0.0


def test_score_head_fire_zero_above_max_distance():
    five_hops_from_fire = (0, 0)
    s = _state_with_fire(ignition=(5, 5))
    assert score_head_fire(s, five_hops_from_fire, max_distance=3) == 0.0


def test_score_head_fire_finite_within_window():
    two_hops_from_fire = (5, 3)
    assert math.isfinite(score_head_fire(_state_with_fire(ignition=(5, 5)), two_hops_from_fire))


def test_score_head_fire_downwind_scores_higher_than_crosswind():
    s = _state_with_fire(ignition=(5, 5))
    downwind = (7, 5)
    crosswind = (5, 3)
    assert score_head_fire(s, downwind) > score_head_fire(s, crosswind)


def test_score_fire_run_zero_when_no_fire():
    assert score_fire_run(_state_no_fire(), (5, 5)) == 0.0


def test_score_fire_run_zero_below_min_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_fire_run(s, (5, 4)) == 0.0


def test_score_fire_run_zero_above_max_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_fire_run(s, (0, 0), max_distance=3) == 0.0


def test_score_fire_run_finite_within_window():
    s = _state_with_fire(ignition=(5, 5))
    assert math.isfinite(score_fire_run(s, (7, 5)))


def test_score_fire_run_downwind_scores_higher_than_crosswind():
    s = _state_with_fire(ignition=(5, 5))
    downwind = (7, 5)
    crosswind = (5, 3)
    assert score_fire_run(s, downwind) > score_fire_run(s, crosswind)


def test_score_fire_run_higher_fuel_scores_higher():
    s = _state_with_fire(ignition=(5, 5))
    high_fuel, low_fuel = (7, 4), (7, 6)
    s.fuel[high_fuel] = 0.9
    s.fuel[low_fuel] = 0.1
    assert score_fire_run(s, high_fuel) > score_fire_run(s, low_fuel)


def test_score_anchor_flank_zero_when_no_fire():
    assert score_anchor_flank(_state_no_fire(), (5, 5)) == 0.0


def test_score_anchor_flank_zero_below_min_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_anchor_flank(s, (5, 4)) == 0.0


def test_score_anchor_flank_zero_above_max_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_anchor_flank(s, (0, 0), max_distance=3) == 0.0


def test_score_anchor_flank_adjacent_to_barrier_scores_higher():
    s = _state_with_fire(ignition=(5, 5))
    s.state[2, 5] = NodeState.TREATED
    anchored, isolated = (3, 5), (5, 3)
    s.fuel[anchored] = s.fuel[isolated] = 0.5
    assert score_anchor_flank(s, anchored) > score_anchor_flank(s, isolated)


def test_score_bottleneck_zero_when_no_fire():
    s = _state_no_fire()
    precompute_reachable_unburned_area(s)
    assert score_bottleneck(s, (5, 5)) == 0.0


def test_score_bottleneck_zero_below_min_distance():
    s = _state_with_fire(ignition=(5, 5))
    precompute_reachable_unburned_area(s)
    assert score_bottleneck(s, (5, 4)) == 0.0


def test_score_bottleneck_zero_above_max_distance():
    s = _state_with_fire(ignition=(5, 5))
    precompute_reachable_unburned_area(s)
    assert score_bottleneck(s, (0, 0), max_distance=3) == 0.0


def test_score_bottleneck_positive_within_window():
    s = _state_with_fire(ignition=(5, 5))
    precompute_reachable_unburned_area(s)
    s.fuel[5, 3] = 0.9
    assert score_bottleneck(s, (5, 3)) > 0.0


def test_score_uphill_interception_zero_when_no_fire():
    assert score_uphill_interception(_state_no_fire(), (5, 5)) == 0.0


def test_score_uphill_interception_zero_below_min_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_uphill_interception(s, (5, 4)) == 0.0


def test_score_uphill_interception_zero_above_max_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_uphill_interception(s, (0, 0), max_distance=3) == 0.0


def test_score_uphill_interception_uphill_scores_higher_than_downhill():
    s = _state_with_fire(ignition=(5, 5))
    s.elevation[5, 5] = 0.5
    uphill, downhill = (3, 5), (7, 5)
    s.elevation[uphill] = 1.0
    s.elevation[downhill] = 0.0
    assert score_uphill_interception(s, uphill) > score_uphill_interception(s, downhill)


def test_score_flank_attack_zero_when_no_fire():
    assert score_flank_attack(_state_no_fire(), (5, 5)) == 0.0


def test_score_flank_attack_zero_below_min_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_flank_attack(s, (5, 4)) == 0.0


def test_score_flank_attack_zero_above_max_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_flank_attack(s, (0, 0), max_distance=3) == 0.0


def test_score_flank_attack_crosswind_scores_higher_than_downwind():
    s = _state_with_fire(ignition=(5, 5))
    crosswind, downwind = (5, 3), (7, 5)
    s.fuel[crosswind] = s.fuel[downwind] = 0.9
    assert score_flank_attack(s, crosswind) > score_flank_attack(s, downwind)


def test_score_composite_threat_zero_when_no_fire():
    assert score_composite_threat(_state_no_fire(), (5, 5)) == 0.0


def test_score_composite_threat_zero_below_min_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_composite_threat(s, (5, 4)) == 0.0


def test_score_composite_threat_zero_above_max_distance():
    s = _state_with_fire(ignition=(5, 5))
    assert score_composite_threat(s, (0, 0), max_distance=3) == 0.0


def test_score_composite_threat_downwind_scores_higher_than_upwind():
    s = _state_with_fire(ignition=(5, 5))
    downwind, upwind = (7, 5), (3, 5)
    s.fuel[downwind] = s.fuel[upwind] = 0.9
    s.slope[downwind] = s.slope[upwind] = 0.5
    assert score_composite_threat(s, downwind) > score_composite_threat(s, upwind)


def _state_graph_baselines(ignition=(5, 5)):
    s = _state_with_fire(ignition=ignition)
    precompute_betweenness_static(s)
    precompute_betweenness_dynamic(s)
    precompute_mtt_pathway_count(s)
    precompute_neighbourhood_maps(s)
    return s


def test_score_betweenness_static_returns_finite():
    s = _state_graph_baselines()
    assert math.isfinite(score_betweenness_static(s, (3, 3)))


def test_score_betweenness_static_anchored_at_least_pure():
    s = _state_graph_baselines()
    s.state[2, 2] = NodeState.TREATED
    precompute_neighbourhood_maps(s)
    assert score_betweenness_static_anchored(s, (3, 3)) >= score_betweenness_static(s, (3, 3))


def test_score_betweenness_dynamic_returns_finite():
    s = _state_graph_baselines()
    assert math.isfinite(score_betweenness_dynamic(s, (3, 3)))


def test_score_betweenness_dynamic_anchored_at_least_pure():
    s = _state_graph_baselines()
    s.state[2, 2] = NodeState.TREATED
    precompute_neighbourhood_maps(s)
    assert score_betweenness_dynamic_anchored(s, (3, 3)) >= score_betweenness_dynamic(s, (3, 3))


def test_score_mtt_pathway_returns_finite():
    s = _state_graph_baselines()
    assert math.isfinite(score_mtt_pathway(s, (3, 3)))


def test_score_mtt_pathway_anchored_at_least_pure():
    s = _state_graph_baselines()
    s.state[2, 2] = NodeState.TREATED
    precompute_neighbourhood_maps(s)
    assert score_mtt_pathway_anchored(s, (3, 3)) >= score_mtt_pathway(s, (3, 3))


def test_score_frontier_protect_zero_without_burning_neighbour():
    s = _state_graph_baselines()
    assert score_frontier_protect(s, (0, 0)) == 0.0


def test_score_frontier_protect_positive_with_burning_neighbour():
    s = _state_graph_baselines(ignition=(5, 5))
    assert score_frontier_protect(s, (5, 4)) > 0.0


def test_score_frontier_protect_anchored_zero_without_burning_neighbour():
    s = _state_graph_baselines()
    s.state[0, 1] = NodeState.TREATED
    precompute_neighbourhood_maps(s)
    assert score_frontier_protect_anchored(s, (0, 0)) == 0.0
