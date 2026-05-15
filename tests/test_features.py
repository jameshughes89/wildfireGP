from wildfireGP.features import (
    burnable_distance_to_fire,
    burning_neighbour_count,
    burning_two_hop_count,
    distance_to_fire,
    elevation,
    elevation_delta_to_fire,
    fuel_level,
    has_treated_neighbour,
    mean_neighbour_elevation,
    mean_neighbour_fuel,
    precompute_burnable_fire_map,
    precompute_fire_map,
    precompute_reachable_unburned_area,
    precompute_state_counts,
    reachable_unburned_area,
    slope,
    total_burned,
    total_burning,
    total_treated,
    total_unburned,
    unburnable_neighbour_count,
    unburned_neighbour_count,
    wind_fire_alignment,
)
from wildfireGP.network import (
    ELEVATION,
    FUEL,
    SLOPE,
    STATE,
    TERRAIN,
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NODE = (1, 1)


def _graph():
    return create_grid(3, 3, seed=0)


def _graph_env():
    g = _graph()
    set_wind(g, speed=15.0, direction=90.0)
    set_fuel_moisture(g, moisture=0.2)
    return g


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def test_node_getters_return_stored_values():
    g = _graph()
    assert fuel_level(g, _NODE) == g.nodes[_NODE][FUEL]
    assert elevation(g, _NODE) == g.nodes[_NODE][ELEVATION]
    assert slope(g, _NODE) == g.nodes[_NODE][SLOPE]


def test_mean_neighbour_elevation_returns_average_of_neighbour_elevation():
    g = _graph()
    g.nodes[(0, 0)][ELEVATION] = 0.2
    g.nodes[(0, 1)][ELEVATION] = 0.2
    g.nodes[(0, 2)][ELEVATION] = 0.4
    g.nodes[(1, 0)][ELEVATION] = 0.4
    g.nodes[(1, 2)][ELEVATION] = 0.6
    g.nodes[(2, 0)][ELEVATION] = 0.6
    g.nodes[(2, 1)][ELEVATION] = 0.8
    g.nodes[(2, 2)][ELEVATION] = 0.8
    assert mean_neighbour_elevation(g, _NODE) == 0.5


def test_mean_neighbour_elevation_uses_available_neighbours_for_edge_node():
    g = _graph()
    g.nodes[(0, 1)][ELEVATION] = 0.2
    g.nodes[(1, 0)][ELEVATION] = 0.8
    g.nodes[(1, 1)][ELEVATION] = 0.5
    assert mean_neighbour_elevation(g, (0, 0)) == 0.5


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------


def test_mean_neighbour_fuel_returns_average_of_neighbour_fuel():
    g = _graph()
    g.nodes[(0, 0)][FUEL] = 0.2
    g.nodes[(0, 1)][FUEL] = 0.2
    g.nodes[(0, 2)][FUEL] = 0.4
    g.nodes[(1, 0)][FUEL] = 0.4
    g.nodes[(1, 2)][FUEL] = 0.6
    g.nodes[(2, 0)][FUEL] = 0.6
    g.nodes[(2, 1)][FUEL] = 0.8
    g.nodes[(2, 2)][FUEL] = 0.8
    assert mean_neighbour_fuel(g, _NODE) == 0.5


def test_mean_neighbour_fuel_uses_available_neighbours_for_edge_node():
    g = _graph()
    g.nodes[(0, 1)][FUEL] = 0.2
    g.nodes[(1, 0)][FUEL] = 0.8
    g.nodes[(1, 1)][FUEL] = 0.5
    assert mean_neighbour_fuel(g, (0, 0)) == 0.5


def test_burning_neighbour_count_zero_with_no_fire():
    g = _graph()
    assert burning_neighbour_count(g, _NODE) == 0


def test_burning_neighbour_count_counts_burning_neighbours():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    g.nodes[(1, 0)][STATE] = NodeState.BURNING
    assert burning_neighbour_count(g, _NODE) == 2


def test_burning_two_hop_count_counts_nodes_exactly_two_hops_away():
    g = create_grid(5, 5, seed=0)
    g.nodes[(0, 2)][STATE] = NodeState.BURNING
    g.nodes[(2, 0)][STATE] = NodeState.BURNING
    assert burning_two_hop_count(g, (2, 2)) == 2


def test_burning_two_hop_count_zero_with_no_fire():
    g = create_grid(5, 5, seed=0)
    assert burning_two_hop_count(g, (2, 2)) == 0


def test_burning_two_hop_count_excludes_immediate_neighbours():
    g = create_grid(5, 5, seed=0)
    g.nodes[(1, 2)][STATE] = NodeState.BURNING
    g.nodes[(0, 2)][STATE] = NodeState.BURNING
    assert burning_two_hop_count(g, (2, 2)) == 1


def test_burning_two_hop_count_deduplicates_nodes_reached_by_multiple_paths():
    g = create_grid(5, 5, seed=0)
    g.nodes[(0, 2)][STATE] = NodeState.BURNING  # reachable from (2, 2) via (1, 1), (1, 2), and (1, 3)
    assert burning_two_hop_count(g, (2, 2)) == 1


def test_burning_two_hop_count_uses_available_two_hop_nodes_for_edge_node():
    g = create_grid(5, 5, seed=0)
    g.nodes[(0, 2)][STATE] = NodeState.BURNING
    g.nodes[(2, 0)][STATE] = NodeState.BURNING
    assert burning_two_hop_count(g, (0, 0)) == 2


def test_unburned_neighbour_count_all_unburned_by_default():
    g = _graph()
    assert unburned_neighbour_count(g, _NODE) == 8


def test_unburned_neighbour_count_excludes_burning():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    assert unburned_neighbour_count(g, _NODE) == 7


def test_unburnable_neighbour_count_includes_rock():
    g = _graph()
    g.nodes[(0, 1)][TERRAIN] = TerrainType.ROCK
    assert unburnable_neighbour_count(g, _NODE) == 1


def test_unburnable_neighbour_count_combines_all_types():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNED
    g.nodes[(1, 0)][STATE] = NodeState.TREATED
    g.nodes[(1, 2)][TERRAIN] = TerrainType.WATER
    assert unburnable_neighbour_count(g, _NODE) == 3


def test_unburnable_neighbour_count_zero_when_all_unburned_land():
    g = _graph()
    assert unburnable_neighbour_count(g, _NODE) == 0


def test_has_treated_neighbour_zero_when_no_treatments():
    g = _graph()
    assert has_treated_neighbour(g, _NODE) == 0.0


def test_has_treated_neighbour_one_when_any_neighbour_treated():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.TREATED
    assert has_treated_neighbour(g, _NODE) == 1.0


def test_has_treated_neighbour_one_regardless_of_count():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.TREATED
    g.nodes[(1, 0)][STATE] = NodeState.TREATED
    assert has_treated_neighbour(g, _NODE) == 1.0


def test_has_treated_neighbour_excludes_burned():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNED
    assert has_treated_neighbour(g, _NODE) == 0.0


def test_has_treated_neighbour_excludes_burning():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    assert has_treated_neighbour(g, _NODE) == 0.0


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------


def test_precompute_fire_map_maps_burning_node_to_itself():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert g.graph["nearest_fire"][_NODE] == _NODE


def test_precompute_fire_map_maps_neighbour_to_fire():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert g.graph["nearest_fire"][(0, 1)] == _NODE


def test_precompute_fire_map_leaves_unreachable_nodes_absent_when_no_fire():
    g = _graph()
    precompute_fire_map(g)
    assert g.graph["nearest_fire"] == {}


def test_distance_to_fire_returns_inf_with_no_fire():
    g = _graph()
    precompute_fire_map(g)
    assert distance_to_fire(g, _NODE) == float("inf")


def test_distance_to_fire_zero_when_node_is_burning():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert distance_to_fire(g, _NODE) == 0


def test_distance_to_fire_hop_count():
    g = create_grid(5, 5, seed=0)
    g.nodes[(4, 4)][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert distance_to_fire(g, (0, 0)) == 4


def test_distance_to_fire_nearest_when_multiple_burning():
    g = create_grid(5, 5, seed=0)
    g.nodes[(0, 4)][STATE] = NodeState.BURNING
    g.nodes[(2, 2)][STATE] = NodeState.BURNING
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert distance_to_fire(g, (0, 0)) == 1


def test_wind_fire_alignment_zero_with_no_fire():
    g = _graph_env()
    precompute_fire_map(g)
    assert wind_fire_alignment(g, _NODE) == 0.0


def test_wind_fire_alignment_zero_when_node_is_burning():
    g = _graph_env()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert wind_fire_alignment(g, _NODE) == 0.0


def test_wind_fire_alignment_positive_one_when_directly_downwind():
    g = create_grid(5, 5, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    g.nodes[(0, 2)][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert abs(wind_fire_alignment(g, (4, 2)) - 1.0) < 1e-9


def test_wind_fire_alignment_negative_one_when_directly_upwind():
    g = create_grid(5, 5, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    g.nodes[(4, 2)][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert abs(wind_fire_alignment(g, (0, 2)) - (-1.0)) < 1e-9


def test_wind_fire_alignment_zero_when_crosswind():
    g = create_grid(5, 5, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    g.nodes[(2, 4)][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert abs(wind_fire_alignment(g, (2, 0))) < 1e-9


def test_elevation_delta_to_fire_zero_with_no_fire():
    g = _graph()
    precompute_fire_map(g)
    assert elevation_delta_to_fire(g, _NODE) == 0.0


def test_elevation_delta_to_fire_zero_when_node_is_burning():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    precompute_fire_map(g)
    assert elevation_delta_to_fire(g, _NODE) == 0.0


def test_elevation_delta_to_fire_positive_when_node_higher_than_fire():
    g = create_grid(5, 5, seed=0)
    fire_node = (2, 0)
    test_node = (2, 4)
    g.nodes[fire_node][STATE] = NodeState.BURNING
    g.nodes[fire_node][ELEVATION] = 0.2
    g.nodes[test_node][ELEVATION] = 0.8
    precompute_fire_map(g)
    assert elevation_delta_to_fire(g, test_node) > 0.0


def test_elevation_delta_to_fire_negative_when_node_lower_than_fire():
    g = create_grid(5, 5, seed=0)
    fire_node = (2, 0)
    test_node = (2, 4)
    g.nodes[fire_node][STATE] = NodeState.BURNING
    g.nodes[fire_node][ELEVATION] = 0.8
    g.nodes[test_node][ELEVATION] = 0.2
    precompute_fire_map(g)
    assert elevation_delta_to_fire(g, test_node) < 0.0


def test_elevation_delta_to_fire_correct_value():
    g = create_grid(3, 3, seed=0)
    g.nodes[(0, 0)][STATE] = NodeState.BURNING
    g.nodes[(0, 0)][ELEVATION] = 0.3
    g.nodes[(2, 2)][ELEVATION] = 0.7
    precompute_fire_map(g)
    assert abs(elevation_delta_to_fire(g, (2, 2)) - 0.4) < 1e-9


def test_burnable_distance_to_fire_returns_inf_with_no_fire():
    g = _graph()
    precompute_burnable_fire_map(g)
    assert burnable_distance_to_fire(g, _NODE) == float("inf")


def test_burnable_distance_to_fire_zero_when_node_is_burning():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    precompute_burnable_fire_map(g)
    assert burnable_distance_to_fire(g, _NODE) == 0


def test_burnable_distance_to_fire_one_for_burnable_neighbour():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    precompute_burnable_fire_map(g)
    assert burnable_distance_to_fire(g, (0, 1)) == 1


def test_burnable_distance_to_fire_inf_when_surrounded_by_unburnable():
    g = _graph()
    g.nodes[(0, 0)][STATE] = NodeState.BURNING
    g.nodes[(0, 1)][STATE] = NodeState.BURNED
    g.nodes[(1, 0)][STATE] = NodeState.BURNED
    g.nodes[(1, 1)][STATE] = NodeState.BURNED
    precompute_burnable_fire_map(g)
    assert burnable_distance_to_fire(g, (2, 2)) == float("inf")


def test_burnable_distance_differs_from_chebyshev_when_path_blocked():
    g = _graph()
    g.nodes[(0, 0)][STATE] = NodeState.BURNING
    g.nodes[(0, 1)][STATE] = NodeState.BURNED
    g.nodes[(1, 0)][STATE] = NodeState.BURNED
    g.nodes[(1, 1)][STATE] = NodeState.BURNED
    precompute_fire_map(g)
    precompute_burnable_fire_map(g)
    assert distance_to_fire(g, (2, 2)) < float("inf")
    assert burnable_distance_to_fire(g, (2, 2)) == float("inf")


# ---------------------------------------------------------------------------
# Reachable unburned area
# ---------------------------------------------------------------------------


def test_reachable_unburned_area_all_unburned_returns_grid_size():
    g = _graph()
    precompute_reachable_unburned_area(g)
    assert reachable_unburned_area(g, _NODE) == 9.0


def test_reachable_unburned_area_single_isolated_node():
    g = _graph()
    for n in g.nodes:
        if n != _NODE:
            g.nodes[n][STATE] = NodeState.BURNED
    precompute_reachable_unburned_area(g)
    assert reachable_unburned_area(g, _NODE) == 1.0


def test_reachable_unburned_area_two_components():
    # Burn the middle column, leaving two separate unburned regions:
    # col 0 (3 nodes) and col 2 (3 nodes)
    g = _graph()
    for row in range(3):
        g.nodes[(row, 1)][STATE] = NodeState.BURNED
    precompute_reachable_unburned_area(g)
    assert reachable_unburned_area(g, (0, 0)) == 3.0
    assert reachable_unburned_area(g, (0, 2)) == 3.0


def test_reachable_unburned_area_fire_pocket():
    # Pocket: (1,1) is unburned, surrounded by burning/burned neighbours — area = 1
    g = _graph()
    for n in g.neighbors(_NODE):
        g.nodes[n][STATE] = NodeState.BURNED
    precompute_reachable_unburned_area(g)
    assert reachable_unburned_area(g, _NODE) == 1.0


def test_reachable_unburned_area_returns_zero_for_burned_node():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNED
    precompute_reachable_unburned_area(g)
    assert reachable_unburned_area(g, _NODE) == 0.0


def test_reachable_unburned_area_returns_zero_for_burning_node():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    precompute_reachable_unburned_area(g)
    assert reachable_unburned_area(g, _NODE) == 0.0


def test_reachable_unburned_area_water_node_returns_zero():
    g = _graph()
    g.nodes[_NODE][TERRAIN] = TerrainType.WATER
    precompute_reachable_unburned_area(g)
    assert reachable_unburned_area(g, _NODE) == 0.0


def test_reachable_unburned_area_pocket_smaller_than_open_front():
    # Burn middle column and the bottom two nodes of col 0, isolating (0,0) as a pocket of size 1.
    # Right column (0,2)-(1,2)-(2,2) remains connected (area 3).
    g = _graph()
    for row in range(3):
        g.nodes[(row, 1)][STATE] = NodeState.BURNED
    g.nodes[(1, 0)][STATE] = NodeState.BURNED
    g.nodes[(2, 0)][STATE] = NodeState.BURNED
    precompute_reachable_unburned_area(g)
    pocket = reachable_unburned_area(g, (0, 0))
    open_front = reachable_unburned_area(g, (0, 2))
    assert pocket == 1.0
    assert open_front == 3.0
    assert pocket < open_front


# ---------------------------------------------------------------------------
# Whole-graph state
# ---------------------------------------------------------------------------


def test_total_state_counts():
    g = _graph()
    g.nodes[(0, 0)][STATE] = NodeState.BURNING
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    g.nodes[(1, 0)][STATE] = NodeState.BURNED
    g.nodes[(1, 1)][STATE] = NodeState.TREATED
    precompute_state_counts(g)
    assert total_burning(g) == 2
    assert total_burned(g) == 1
    assert total_treated(g) == 1
    assert total_unburned(g) == 5


def test_totals_sum_to_node_count():
    g = _graph()
    g.nodes[(0, 0)][STATE] = NodeState.BURNING
    g.nodes[(0, 1)][STATE] = NodeState.BURNED
    g.nodes[(0, 2)][STATE] = NodeState.TREATED
    precompute_state_counts(g)
    total = total_burning(g) + total_burned(g) + total_unburned(g) + total_treated(g)
    assert total == len(g.nodes)
