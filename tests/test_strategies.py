import math

from wildfireGP.features import precompute_burnable_fire_map, precompute_fire_map
from wildfireGP.network import (
    NodeState,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.spread import MAX_BURN_STEPS
from wildfireGP.strategies import (
    random_score,
    score_by_burning_neighbors,
    score_by_fire_proximity,
    score_by_fuel,
    score_fire_run,
    score_head_fire,
    score_indirect_attack,
    score_ridgeline,
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
