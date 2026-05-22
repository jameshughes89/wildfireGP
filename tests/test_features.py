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
    precompute_burning_two_hop_map,
    precompute_fire_distance_map,
    precompute_fire_map,
    precompute_neighbourhood_maps,
    precompute_reachable_unburned_area,
    precompute_state_counts,
    reachable_unburned_area,
    slope,
    total_burned,
    total_burning,
    total_treated,
    total_unburned,
    treated_neighbour_count,
    unburnable_neighbour_count,
    unburned_neighbour_count,
    update_neighbourhood_maps_after_treatment,
    wind_fire_alignment,
)
from wildfireGP.network import (
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)

_NODE = (1, 1)


def _state():
    return create_grid(3, 3, seed=0)


def _state_env():
    s = _state()
    set_wind(s, speed=15.0, direction=90.0)
    set_fuel_moisture(s, moisture=0.2)
    return s


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def test_node_getters_return_stored_values():
    s = _state()
    assert fuel_level(s, _NODE) == float(s.fuel[_NODE])
    assert elevation(s, _NODE) == float(s.elevation[_NODE])
    assert slope(s, _NODE) == float(s.slope[_NODE])


def test_mean_neighbour_elevation_returns_average_of_neighbour_elevation():
    s = _state()
    s.elevation[0, 0] = 0.2
    s.elevation[0, 1] = 0.2
    s.elevation[0, 2] = 0.4
    s.elevation[1, 0] = 0.4
    s.elevation[1, 2] = 0.6
    s.elevation[2, 0] = 0.6
    s.elevation[2, 1] = 0.8
    s.elevation[2, 2] = 0.8
    assert abs(mean_neighbour_elevation(s, _NODE) - 0.5) < 1e-6


def test_mean_neighbour_elevation_uses_available_neighbours_for_edge_node():
    s = _state()
    s.elevation[0, 1] = 0.2
    s.elevation[1, 0] = 0.8
    s.elevation[1, 1] = 0.5
    assert abs(mean_neighbour_elevation(s, (0, 0)) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------


def test_mean_neighbour_fuel_returns_average_of_neighbour_fuel():
    s = _state()
    s.fuel[0, 0] = 0.2
    s.fuel[0, 1] = 0.2
    s.fuel[0, 2] = 0.4
    s.fuel[1, 0] = 0.4
    s.fuel[1, 2] = 0.6
    s.fuel[2, 0] = 0.6
    s.fuel[2, 1] = 0.8
    s.fuel[2, 2] = 0.8
    assert abs(mean_neighbour_fuel(s, _NODE) - 0.5) < 1e-6


def test_mean_neighbour_fuel_uses_available_neighbours_for_edge_node():
    s = _state()
    s.fuel[0, 1] = 0.2
    s.fuel[1, 0] = 0.8
    s.fuel[1, 1] = 0.5
    assert abs(mean_neighbour_fuel(s, (0, 0)) - 0.5) < 1e-6


def test_burning_neighbour_count_zero_with_no_fire():
    assert burning_neighbour_count(_state(), _NODE) == 0


def test_burning_neighbour_count_counts_burning_neighbours():
    s = _state()
    s.state[0, 1] = NodeState.BURNING
    s.state[1, 0] = NodeState.BURNING
    assert burning_neighbour_count(s, _NODE) == 2


def test_burning_two_hop_count_counts_cells_exactly_two_hops_away():
    s = create_grid(5, 5, seed=0)
    s.state[0, 2] = NodeState.BURNING
    s.state[2, 0] = NodeState.BURNING
    assert burning_two_hop_count(s, (2, 2)) == 2


def test_burning_two_hop_count_zero_with_no_fire():
    s = create_grid(5, 5, seed=0)
    assert burning_two_hop_count(s, (2, 2)) == 0


def test_burning_two_hop_count_excludes_immediate_neighbours():
    s = create_grid(5, 5, seed=0)
    s.state[1, 2] = NodeState.BURNING
    s.state[0, 2] = NodeState.BURNING
    assert burning_two_hop_count(s, (2, 2)) == 1


def test_burning_two_hop_count_deduplicates_cells_reached_by_multiple_paths():
    s = create_grid(5, 5, seed=0)
    s.state[0, 2] = NodeState.BURNING
    assert burning_two_hop_count(s, (2, 2)) == 1


def test_burning_two_hop_count_uses_available_two_hop_cells_for_edge_node():
    s = create_grid(5, 5, seed=0)
    s.state[0, 2] = NodeState.BURNING
    s.state[2, 0] = NodeState.BURNING
    assert burning_two_hop_count(s, (0, 0)) == 2


def test_precompute_burning_two_hop_map_matches_direct_feature_reads():
    s = create_grid(5, 5, seed=0)
    s.state[0, 2] = NodeState.BURNING
    s.state[2, 0] = NodeState.BURNING
    s.state[2, 2] = NodeState.BURNING
    direct = {
        (2, 2): burning_two_hop_count(s, (2, 2)),
        (0, 0): burning_two_hop_count(s, (0, 0)),
        (4, 4): burning_two_hop_count(s, (4, 4)),
    }
    precompute_burning_two_hop_map(s)
    assert burning_two_hop_count(s, (2, 2)) == direct[(2, 2)]
    assert burning_two_hop_count(s, (0, 0)) == direct[(0, 0)]
    assert burning_two_hop_count(s, (4, 4)) == direct[(4, 4)]


def test_unburned_neighbour_count_all_unburned_by_default():
    assert unburned_neighbour_count(_state(), _NODE) == 8


def test_unburned_neighbour_count_excludes_burning():
    s = _state()
    s.state[0, 1] = NodeState.BURNING
    assert unburned_neighbour_count(s, _NODE) == 7


def test_unburnable_neighbour_count_includes_rock():
    s = _state()
    s.terrain[0, 1] = TerrainType.ROCK
    assert unburnable_neighbour_count(s, _NODE) == 1


def test_unburnable_neighbour_count_combines_all_types():
    s = _state()
    s.state[0, 1] = NodeState.BURNED
    s.state[1, 0] = NodeState.TREATED
    s.terrain[1, 2] = TerrainType.WATER
    assert unburnable_neighbour_count(s, _NODE) == 3


def test_unburnable_neighbour_count_zero_when_all_unburned_land():
    assert unburnable_neighbour_count(_state(), _NODE) == 0


def test_has_treated_neighbour_zero_when_no_treatments():
    assert has_treated_neighbour(_state(), _NODE) == 0.0


def test_has_treated_neighbour_one_when_any_neighbour_treated():
    s = _state()
    s.state[0, 1] = NodeState.TREATED
    assert has_treated_neighbour(s, _NODE) == 1.0


def test_has_treated_neighbour_one_regardless_of_count():
    s = _state()
    s.state[0, 1] = NodeState.TREATED
    s.state[1, 0] = NodeState.TREATED
    assert has_treated_neighbour(s, _NODE) == 1.0


def test_has_treated_neighbour_excludes_burned():
    s = _state()
    s.state[0, 1] = NodeState.BURNED
    assert has_treated_neighbour(s, _NODE) == 0.0


def test_has_treated_neighbour_excludes_burning():
    s = _state()
    s.state[0, 1] = NodeState.BURNING
    assert has_treated_neighbour(s, _NODE) == 0.0


def test_treated_neighbour_count_zero_when_no_treatments():
    assert treated_neighbour_count(_state(), _NODE) == 0


def test_treated_neighbour_count_one_when_single_neighbour_treated():
    s = _state()
    s.state[0, 1] = NodeState.TREATED
    assert treated_neighbour_count(s, _NODE) == 1


def test_treated_neighbour_count_reflects_exact_number():
    s = _state()
    s.state[0, 1] = NodeState.TREATED
    s.state[1, 0] = NodeState.TREATED
    s.state[2, 2] = NodeState.TREATED
    assert treated_neighbour_count(s, _NODE) == 3


def test_treated_neighbour_count_excludes_burned():
    s = _state()
    s.state[0, 1] = NodeState.BURNED
    assert treated_neighbour_count(s, _NODE) == 0


def test_treated_neighbour_count_excludes_burning():
    s = _state()
    s.state[0, 1] = NodeState.BURNING
    assert treated_neighbour_count(s, _NODE) == 0


def test_precompute_neighbourhood_maps_matches_direct_feature_reads():
    s = _state()
    s.state[0, 0] = NodeState.BURNING
    s.state[0, 1] = NodeState.TREATED
    s.state[1, 0] = NodeState.BURNED
    s.terrain[1, 2] = TerrainType.WATER
    direct = {
        "mean_elevation": mean_neighbour_elevation(s, _NODE),
        "mean_fuel": mean_neighbour_fuel(s, _NODE),
        "burning": burning_neighbour_count(s, _NODE),
        "unburned": unburned_neighbour_count(s, _NODE),
        "unburnable": unburnable_neighbour_count(s, _NODE),
        "treated_count": treated_neighbour_count(s, _NODE),
        "has_treated": has_treated_neighbour(s, _NODE),
    }
    precompute_neighbourhood_maps(s)
    assert abs(mean_neighbour_elevation(s, _NODE) - direct["mean_elevation"]) < 1e-6
    assert abs(mean_neighbour_fuel(s, _NODE) - direct["mean_fuel"]) < 1e-6
    assert burning_neighbour_count(s, _NODE) == direct["burning"]
    assert unburned_neighbour_count(s, _NODE) == direct["unburned"]
    assert unburnable_neighbour_count(s, _NODE) == direct["unburnable"]
    assert treated_neighbour_count(s, _NODE) == direct["treated_count"]
    assert has_treated_neighbour(s, _NODE) == direct["has_treated"]


def test_update_neighbourhood_maps_after_treatment_keeps_counts_in_sync():
    s = _state()
    precompute_neighbourhood_maps(s)
    treated_node = (0, 1)
    s.state[treated_node] = NodeState.TREATED
    update_neighbourhood_maps_after_treatment(s, treated_node)
    assert treated_neighbour_count(s, _NODE) == 1
    assert has_treated_neighbour(s, _NODE) == 1.0
    assert unburned_neighbour_count(s, _NODE) == 7
    assert unburnable_neighbour_count(s, _NODE) == 1


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------


def test_precompute_fire_map_maps_burning_cell_to_itself():
    s = _state()
    s.state[_NODE] = NodeState.BURNING
    precompute_fire_map(s)
    assert s.nearest_fire[_NODE] == _NODE


def test_precompute_fire_map_maps_neighbour_to_fire():
    s = _state()
    s.state[_NODE] = NodeState.BURNING
    precompute_fire_map(s)
    assert s.nearest_fire[(0, 1)] == _NODE


def test_precompute_fire_map_leaves_unreachable_cells_absent_when_no_fire():
    s = _state()
    precompute_fire_map(s)
    assert s.nearest_fire == {}


def test_distance_to_fire_returns_inf_with_no_fire():
    s = _state()
    precompute_fire_distance_map(s)
    assert distance_to_fire(s, _NODE) == float("inf")


def test_distance_to_fire_zero_when_cell_is_burning():
    s = _state()
    s.state[_NODE] = NodeState.BURNING
    precompute_fire_distance_map(s)
    assert distance_to_fire(s, _NODE) == 0


def test_distance_to_fire_hop_count():
    s = create_grid(5, 5, seed=0)
    s.state[4, 4] = NodeState.BURNING
    precompute_fire_distance_map(s)
    assert distance_to_fire(s, (0, 0)) == 4


def test_distance_to_fire_nearest_when_multiple_burning():
    s = create_grid(5, 5, seed=0)
    s.state[0, 4] = NodeState.BURNING
    s.state[2, 2] = NodeState.BURNING
    s.state[0, 1] = NodeState.BURNING
    precompute_fire_distance_map(s)
    assert distance_to_fire(s, (0, 0)) == 1


def test_precompute_fire_map_also_populates_distance_map():
    s = _state()
    s.state[_NODE] = NodeState.BURNING
    precompute_fire_map(s)
    assert distance_to_fire(s, _NODE) == 0.0


def test_wind_fire_alignment_zero_with_no_fire():
    s = _state_env()
    precompute_fire_map(s)
    assert wind_fire_alignment(s, _NODE) == 0.0


def test_wind_fire_alignment_zero_when_cell_is_burning():
    s = _state_env()
    s.state[_NODE] = NodeState.BURNING
    precompute_fire_map(s)
    assert wind_fire_alignment(s, _NODE) == 0.0


def test_wind_fire_alignment_positive_one_when_directly_downwind():
    s = create_grid(5, 5, seed=0)
    set_wind(s, speed=10.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.1)
    s.state[0, 2] = NodeState.BURNING
    precompute_fire_map(s)
    assert abs(wind_fire_alignment(s, (4, 2)) - 1.0) < 1e-9


def test_wind_fire_alignment_negative_one_when_directly_upwind():
    s = create_grid(5, 5, seed=0)
    set_wind(s, speed=10.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.1)
    s.state[4, 2] = NodeState.BURNING
    precompute_fire_map(s)
    assert abs(wind_fire_alignment(s, (0, 2)) - (-1.0)) < 1e-9


def test_wind_fire_alignment_zero_when_crosswind():
    s = create_grid(5, 5, seed=0)
    set_wind(s, speed=10.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.1)
    s.state[2, 4] = NodeState.BURNING
    precompute_fire_map(s)
    assert abs(wind_fire_alignment(s, (2, 0))) < 1e-9


def test_elevation_delta_to_fire_zero_with_no_fire():
    s = _state()
    precompute_fire_map(s)
    assert elevation_delta_to_fire(s, _NODE) == 0.0


def test_elevation_delta_to_fire_zero_when_cell_is_burning():
    s = _state()
    s.state[_NODE] = NodeState.BURNING
    precompute_fire_map(s)
    assert elevation_delta_to_fire(s, _NODE) == 0.0


def test_elevation_delta_to_fire_positive_when_cell_higher_than_fire():
    s = create_grid(5, 5, seed=0)
    fire_node = (2, 0)
    test_node = (2, 4)
    s.state[fire_node] = NodeState.BURNING
    s.elevation[fire_node] = 0.2
    s.elevation[test_node] = 0.8
    precompute_fire_map(s)
    assert elevation_delta_to_fire(s, test_node) > 0.0


def test_elevation_delta_to_fire_negative_when_cell_lower_than_fire():
    s = create_grid(5, 5, seed=0)
    fire_node = (2, 0)
    test_node = (2, 4)
    s.state[fire_node] = NodeState.BURNING
    s.elevation[fire_node] = 0.8
    s.elevation[test_node] = 0.2
    precompute_fire_map(s)
    assert elevation_delta_to_fire(s, test_node) < 0.0


def test_elevation_delta_to_fire_correct_value():
    s = create_grid(3, 3, seed=0)
    s.state[0, 0] = NodeState.BURNING
    s.elevation[0, 0] = 0.3
    s.elevation[2, 2] = 0.7
    precompute_fire_map(s)
    assert abs(elevation_delta_to_fire(s, (2, 2)) - 0.4) < 1e-6


def test_burnable_distance_to_fire_returns_inf_with_no_fire():
    s = _state()
    precompute_burnable_fire_map(s)
    assert burnable_distance_to_fire(s, _NODE) == float("inf")


def test_burnable_distance_to_fire_zero_when_cell_is_burning():
    s = _state()
    s.state[_NODE] = NodeState.BURNING
    precompute_burnable_fire_map(s)
    assert burnable_distance_to_fire(s, _NODE) == 0


def test_burnable_distance_to_fire_one_for_burnable_neighbour():
    s = _state()
    s.state[_NODE] = NodeState.BURNING
    precompute_burnable_fire_map(s)
    assert burnable_distance_to_fire(s, (0, 1)) == 1


def test_burnable_distance_to_fire_inf_when_surrounded_by_unburnable():
    s = _state()
    s.state[0, 0] = NodeState.BURNING
    s.state[0, 1] = NodeState.BURNED
    s.state[1, 0] = NodeState.BURNED
    s.state[1, 1] = NodeState.BURNED
    precompute_burnable_fire_map(s)
    assert burnable_distance_to_fire(s, (2, 2)) == float("inf")


def test_burnable_distance_differs_from_chebyshev_when_path_blocked():
    s = _state()
    s.state[0, 0] = NodeState.BURNING
    s.state[0, 1] = NodeState.BURNED
    s.state[1, 0] = NodeState.BURNED
    s.state[1, 1] = NodeState.BURNED
    precompute_fire_map(s)
    precompute_burnable_fire_map(s)
    assert distance_to_fire(s, (2, 2)) < float("inf")
    assert burnable_distance_to_fire(s, (2, 2)) == float("inf")


def test_burnable_distance_to_fire_inf_when_only_path_is_through_water():
    s = _state()
    s.state[0, 0] = NodeState.BURNING
    for c in range(3):
        s.terrain[1, c] = TerrainType.WATER
    precompute_burnable_fire_map(s)
    assert burnable_distance_to_fire(s, (2, 0)) == float("inf")


def test_burnable_distance_to_fire_inf_when_only_path_is_through_rock():
    s = _state()
    s.state[0, 0] = NodeState.BURNING
    for c in range(3):
        s.terrain[1, c] = TerrainType.ROCK
    precompute_burnable_fire_map(s)
    assert burnable_distance_to_fire(s, (2, 0)) == float("inf")


# ---------------------------------------------------------------------------
# Reachable unburned area
# ---------------------------------------------------------------------------


def test_reachable_unburned_area_all_unburned_returns_grid_size():
    s = _state()
    precompute_reachable_unburned_area(s)
    assert reachable_unburned_area(s, _NODE) == 9.0


def test_reachable_unburned_area_single_isolated_cell():
    s = _state()
    for n in s.nodes():
        if n != _NODE:
            s.state[n] = NodeState.BURNED
    precompute_reachable_unburned_area(s)
    assert reachable_unburned_area(s, _NODE) == 1.0


def test_reachable_unburned_area_two_components():
    s = _state()
    for row in range(3):
        s.state[row, 1] = NodeState.BURNED
    precompute_reachable_unburned_area(s)
    assert reachable_unburned_area(s, (0, 0)) == 3.0
    assert reachable_unburned_area(s, (0, 2)) == 3.0


def test_reachable_unburned_area_fire_pocket():
    s = _state()
    for n in s.neighbours(_NODE):
        s.state[n] = NodeState.BURNED
    precompute_reachable_unburned_area(s)
    assert reachable_unburned_area(s, _NODE) == 1.0


def test_reachable_unburned_area_returns_zero_for_burned_cell():
    s = _state()
    s.state[_NODE] = NodeState.BURNED
    precompute_reachable_unburned_area(s)
    assert reachable_unburned_area(s, _NODE) == 0.0


def test_reachable_unburned_area_returns_zero_for_burning_cell():
    s = _state()
    s.state[_NODE] = NodeState.BURNING
    precompute_reachable_unburned_area(s)
    assert reachable_unburned_area(s, _NODE) == 0.0


def test_reachable_unburned_area_water_cell_returns_zero():
    s = _state()
    s.terrain[_NODE] = TerrainType.WATER
    precompute_reachable_unburned_area(s)
    assert reachable_unburned_area(s, _NODE) == 0.0


def test_reachable_unburned_area_pocket_smaller_than_open_front():
    s = _state()
    for row in range(3):
        s.state[row, 1] = NodeState.BURNED
    s.state[1, 0] = NodeState.BURNED
    s.state[2, 0] = NodeState.BURNED
    precompute_reachable_unburned_area(s)
    pocket = reachable_unburned_area(s, (0, 0))
    open_front = reachable_unburned_area(s, (0, 2))
    assert pocket == 1.0
    assert open_front == 3.0
    assert pocket < open_front


# ---------------------------------------------------------------------------
# Whole-graph state
# ---------------------------------------------------------------------------


def test_total_state_counts():
    s = _state()
    s.state[0, 0] = NodeState.BURNING
    s.state[0, 1] = NodeState.BURNING
    s.state[1, 0] = NodeState.BURNED
    s.state[1, 1] = NodeState.TREATED
    precompute_state_counts(s)
    assert total_burning(s) == 2
    assert total_burned(s) == 1
    assert total_treated(s) == 1
    assert total_unburned(s) == 5


def test_totals_sum_to_cell_count():
    s = _state()
    s.state[0, 0] = NodeState.BURNING
    s.state[0, 1] = NodeState.BURNED
    s.state[0, 2] = NodeState.TREATED
    precompute_state_counts(s)
    total = total_burning(s) + total_burned(s) + total_unburned(s) + total_treated(s)
    assert total == s.rows * s.cols
