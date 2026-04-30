from wildfireGP.network import (
    CELL_SIZE,
    COLS,
    ELEVATION,
    FUEL,
    FUEL_MOISTURE,
    ROWS,
    SLOPE,
    STATE,
    TERRAIN,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
    TerrainType,
    create_grid,
    reset_states,
    set_fuel_moisture,
    set_wind,
)


def test_create_grid_4by5_has_20_nodes():
    assert create_grid(4, 5).number_of_nodes() == 20


def test_create_grid_default_all_nodes_unburned():
    graph = create_grid(4, 5)
    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)


def test_create_grid_default_fuel_in_unit_interval():
    graph = create_grid(10, 10)
    assert all(0.0 <= graph.nodes[n][FUEL] <= 1.0 for n in graph.nodes)


def test_create_grid_default_slope_in_unit_interval():
    graph = create_grid(10, 10)
    assert all(0.0 <= graph.nodes[n][SLOPE] <= 1.0 for n in graph.nodes)


def test_create_grid_default_elevation_in_unit_interval():
    graph = create_grid(10, 10)
    assert all(0.0 <= graph.nodes[n][ELEVATION] <= 1.0 for n in graph.nodes)


def test_create_grid_same_seed_fuel_is_reproducible():
    g1, g2 = create_grid(5, 5, seed=42), create_grid(5, 5, seed=42)
    assert [g1.nodes[n][FUEL] for n in g1.nodes] == [g2.nodes[n][FUEL] for n in g2.nodes]


def test_create_grid_same_seed_slope_is_reproducible():
    g1, g2 = create_grid(5, 5, seed=42), create_grid(5, 5, seed=42)
    assert [g1.nodes[n][SLOPE] for n in g1.nodes] == [g2.nodes[n][SLOPE] for n in g2.nodes]


def test_create_grid_same_seed_elevation_is_reproducible():
    g1, g2 = create_grid(5, 5, seed=42), create_grid(5, 5, seed=42)
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


def test_create_grid_water_fraction_sets_terrain_type_water():
    graph = create_grid(20, 20, water_fraction=0.2, seed=0)
    assert any(graph.nodes[n][TERRAIN] == TerrainType.WATER for n in graph.nodes)


def test_create_grid_rock_fraction_sets_terrain_type_rock():
    graph = create_grid(20, 20, rock_fraction=0.2, seed=0)
    assert any(graph.nodes[n][TERRAIN] == TerrainType.ROCK for n in graph.nodes)


def test_create_grid_water_fraction_sets_fuel_to_zero():
    graph = create_grid(20, 20, water_fraction=0.2, seed=0)
    assert sum(1 for n in graph.nodes if graph.nodes[n][FUEL] == 0.0) > 1


def test_create_grid_rock_fraction_sets_fuel_to_zero():
    graph = create_grid(20, 20, rock_fraction=0.2, seed=0)
    assert sum(1 for n in graph.nodes if graph.nodes[n][FUEL] == 0.0) > 1


def test_create_grid_default_cell_size_is_100():
    assert create_grid(3, 3).graph[CELL_SIZE] == 100.0


def test_create_grid_custom_cell_size_stored_on_graph():
    assert create_grid(3, 3, cell_size_m=30.0).graph[CELL_SIZE] == 30.0


def test_create_grid_rows_stored_on_graph():
    assert create_grid(4, 5).graph[ROWS] == 4


def test_create_grid_cols_stored_on_graph():
    assert create_grid(4, 5).graph[COLS] == 5


def test_create_grid_default_wind_speed_not_stored():
    assert WIND_SPEED not in create_grid(3, 3).graph


def test_create_grid_default_wind_direction_not_stored():
    assert WIND_DIRECTION not in create_grid(3, 3).graph


def test_create_grid_default_fuel_moisture_not_stored():
    assert FUEL_MOISTURE not in create_grid(3, 3).graph


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


def test_reset_states_all_nodes_become_unburned():
    graph = create_grid(3, 3)
    nodes = list(graph.nodes)
    graph.nodes[nodes[0]][STATE] = NodeState.BURNING
    graph.nodes[nodes[1]][STATE] = NodeState.BURNED
    graph.nodes[nodes[2]][STATE] = NodeState.TREATED
    reset_states(graph)
    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)
