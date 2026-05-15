import numpy as np
import pytest

from wildfireGP.network import (
    BURN_TIMER,
    CELL_SIZE,
    ELEVATION,
    FUEL,
    FUEL_MOISTURE,
    SLOPE,
    STATE,
    TERRAIN,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
    TerrainType,
    create_grid,
    reset_states,
    select_ignition_cluster,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)


def test_create_grid_4by5_has_20_nodes():
    assert create_grid(4, 5).number_of_nodes() == 20


def test_create_grid_interior_node_has_8_neighbours():
    assert len(list(create_grid(5, 5).neighbors((2, 2)))) == 8


def test_create_grid_edge_node_has_5_neighbours():
    assert len(list(create_grid(5, 5).neighbors((0, 2)))) == 5


def test_create_grid_corner_node_has_3_neighbours():
    assert len(list(create_grid(5, 5).neighbors((0, 0)))) == 3


def test_create_grid_default_all_nodes_unburned():
    graph = create_grid(4, 5)
    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)


def test_create_grid_default_attributes_in_unit_interval():
    graph = create_grid(10, 10)
    assert all(0.0 <= graph.nodes[n][FUEL] <= 1.0 for n in graph.nodes)
    assert all(0.0 <= graph.nodes[n][SLOPE] <= 1.0 for n in graph.nodes)
    assert all(0.0 <= graph.nodes[n][ELEVATION] <= 1.0 for n in graph.nodes)


def test_create_grid_same_seed_is_reproducible():
    g1, g2 = create_grid(5, 5, seed=42), create_grid(5, 5, seed=42)
    assert [g1.nodes[n][FUEL] for n in g1.nodes] == [g2.nodes[n][FUEL] for n in g2.nodes]
    assert [g1.nodes[n][SLOPE] for n in g1.nodes] == [g2.nodes[n][SLOPE] for n in g2.nodes]
    assert [g1.nodes[n][ELEVATION] for n in g1.nodes] == [g2.nodes[n][ELEVATION] for n in g2.nodes]


def test_create_grid_independent_smoothing_fuel_differs():
    g1 = create_grid(10, 10, terrain_smoothing=1.0, fuel_smoothing=5.0, seed=0)
    g2 = create_grid(10, 10, terrain_smoothing=5.0, fuel_smoothing=1.0, seed=0)
    assert [g1.nodes[n][FUEL] for n in g1.nodes] != [g2.nodes[n][FUEL] for n in g2.nodes]


def test_create_grid_independent_smoothing_slope_differs():
    g1 = create_grid(10, 10, terrain_smoothing=1.0, fuel_smoothing=5.0, seed=0)
    g2 = create_grid(10, 10, terrain_smoothing=5.0, fuel_smoothing=1.0, seed=0)
    assert [g1.nodes[n][SLOPE] for n in g1.nodes] != [g2.nodes[n][SLOPE] for n in g2.nodes]


def test_create_grid_different_seeds_fuel_differs():
    g1, g2 = create_grid(5, 5, seed=0), create_grid(5, 5, seed=1)
    assert [g1.nodes[n][FUEL] for n in g1.nodes] != [g2.nodes[n][FUEL] for n in g2.nodes]


def test_create_grid_default_terrain_type_is_land():
    graph = create_grid(10, 10, seed=0)
    assert all(graph.nodes[n][TERRAIN] == TerrainType.LAND for n in graph.nodes)


def test_create_grid_water_fraction_exact_count():
    rows, cols, fraction = 20, 20, 0.1
    graph = create_grid(rows, cols, water_fraction=fraction, seed=0)
    expected = round(rows * cols * fraction)
    actual = sum(1 for n in graph.nodes if graph.nodes[n][TERRAIN] == TerrainType.WATER)
    assert actual == expected


def test_create_grid_rock_fraction_exact_count():
    rows, cols, fraction = 20, 20, 0.1
    graph = create_grid(rows, cols, rock_fraction=fraction, seed=0)
    expected = round(rows * cols * fraction)
    actual = sum(1 for n in graph.nodes if graph.nodes[n][TERRAIN] == TerrainType.ROCK)
    assert actual == expected


def test_create_grid_rock_overwrites_water_when_both_nonzero():
    graph = create_grid(20, 20, water_fraction=0.5, rock_fraction=0.5, seed=0)
    assert not any(
        graph.nodes[n][TERRAIN] == TerrainType.WATER and graph.nodes[n][TERRAIN] == TerrainType.ROCK
        for n in graph.nodes
    )
    n_rock = sum(1 for n in graph.nodes if graph.nodes[n][TERRAIN] == TerrainType.ROCK)
    assert n_rock == round(20 * 20 * 0.5)


def test_create_grid_custom_cell_size_stored_on_graph():
    assert create_grid(3, 3, cell_size_m=30.0).graph[CELL_SIZE] == 30.0


def test_set_wind_stores_speed_on_graph():
    graph = create_grid(3, 3)
    set_wind(graph, speed=30.0, direction=180.0)
    assert graph.graph[WIND_SPEED] == 30.0


def test_set_wind_stores_direction_on_graph():
    graph = create_grid(3, 3)
    set_wind(graph, speed=30.0, direction=180.0)
    assert graph.graph[WIND_DIRECTION] == 180.0


def test_set_fuel_moisture_stores_value_on_graph():
    graph = create_grid(3, 3)
    set_fuel_moisture(graph, 0.25)
    assert graph.graph[FUEL_MOISTURE] == 0.25


def test_create_grid_default_burn_timer_is_zero():
    graph = create_grid(4, 5)
    assert all(graph.nodes[n][BURN_TIMER] == 0 for n in graph.nodes)


def test_reset_states_all_nodes_become_unburned():
    graph = create_grid(3, 3)
    nodes = list(graph.nodes)
    graph.nodes[nodes[0]][STATE] = NodeState.BURNING
    graph.nodes[nodes[1]][STATE] = NodeState.BURNED
    graph.nodes[nodes[2]][STATE] = NodeState.TREATED
    reset_states(graph)
    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)


def test_reset_states_all_burn_timers_return_to_zero():
    graph = create_grid(3, 3)
    for n in graph.nodes:
        graph.nodes[n][BURN_TIMER] = 3
    reset_states(graph)
    assert all(graph.nodes[n][BURN_TIMER] == 0 for n in graph.nodes)


def test_select_ignition_node_is_burnable():
    graph = create_grid(20, 20, seed=0)
    node = select_ignition_node(graph, np.random.default_rng(0))
    assert graph.nodes[node][TERRAIN] == TerrainType.LAND
    assert graph.nodes[node][FUEL] > 0.0
    assert graph.nodes[node][STATE] == NodeState.UNBURNED


def test_select_ignition_node_is_in_central_region():
    rows, cols = 20, 20
    graph = create_grid(rows, cols, seed=0)
    centre_fraction = 0.5
    margin = (1.0 - centre_fraction) / 2.0
    row_lo, row_hi = int(rows * margin), int(rows * (1.0 - margin))
    col_lo, col_hi = int(cols * margin), int(cols * (1.0 - margin))
    for _ in range(20):
        r, c = select_ignition_node(graph, np.random.default_rng(_), centre_fraction=centre_fraction)
        assert row_lo <= r < row_hi
        assert col_lo <= c < col_hi


def test_select_ignition_node_is_reproducible_with_same_seed():
    graph = create_grid(20, 20, seed=0)
    n1 = select_ignition_node(graph, np.random.default_rng(42))
    n2 = select_ignition_node(graph, np.random.default_rng(42))
    assert n1 == n2


def test_select_ignition_node_falls_back_when_centre_has_no_burnable_nodes():
    graph = create_grid(10, 10, seed=0)
    rows, cols = 10, 10
    margin = (1.0 - 0.5) / 2.0
    for r in range(int(rows * margin), int(rows * (1.0 - margin))):
        for c in range(int(cols * margin), int(cols * (1.0 - margin))):
            graph.nodes[(r, c)][FUEL] = 0.0
            graph.nodes[(r, c)][TERRAIN] = TerrainType.WATER
    node = select_ignition_node(graph, np.random.default_rng(0))
    assert graph.nodes[node][FUEL] > 0.0
    assert graph.nodes[node][TERRAIN] == TerrainType.LAND


def test_select_ignition_node_raises_when_no_burnable_nodes_exist():
    graph = create_grid(5, 5, seed=0)
    for n in graph.nodes:
        graph.nodes[n][FUEL] = 0.0
        graph.nodes[n][TERRAIN] = TerrainType.WATER
    with pytest.raises(ValueError):
        select_ignition_node(graph, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# select_ignition_cluster
# ---------------------------------------------------------------------------


def test_select_ignition_cluster_returns_requested_size():
    graph = create_grid(10, 10, seed=0)
    cluster = select_ignition_cluster(graph, np.random.default_rng(0), size=3)
    assert len(cluster) == 3


def test_select_ignition_cluster_all_nodes_are_burnable():
    graph = create_grid(10, 10, seed=0)
    cluster = select_ignition_cluster(graph, np.random.default_rng(0), size=3)
    for node in cluster:
        assert graph.nodes[node][STATE] == NodeState.UNBURNED
        assert graph.nodes[node][TERRAIN] == TerrainType.LAND
        assert graph.nodes[node][FUEL] > 0.0


def test_select_ignition_cluster_nodes_are_unique():
    graph = create_grid(10, 10, seed=0)
    cluster = select_ignition_cluster(graph, np.random.default_rng(0), size=5)
    assert len(cluster) == len(set(cluster))


def test_select_ignition_cluster_size_one_equals_single_node():
    graph = create_grid(10, 10, seed=0)
    cluster = select_ignition_cluster(graph, np.random.default_rng(0), size=1)
    assert len(cluster) == 1


def test_select_ignition_cluster_capped_by_available_neighbours():
    graph = create_grid(3, 3, seed=0)
    cluster = select_ignition_cluster(graph, np.random.default_rng(0), size=100)
    assert len(cluster) <= graph.number_of_nodes()
