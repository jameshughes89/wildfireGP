from wildfireGP.network import (
    CELL_SIZE,
    ELEVATION,
    FUEL,
    FUEL_MOISTURE,
    SLOPE,
    STATE,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
    create_grid,
    reset_states,
    set_fuel_moisture,
    set_wind,
)

# ---------------------------------------------------------------------------
# create_grid
# ---------------------------------------------------------------------------


def test_create_grid_node_count():
    graph = create_grid(4, 5)
    assert graph.number_of_nodes() == 20


def test_create_grid_all_unburned():
    graph = create_grid(4, 5)
    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)


def test_create_grid_fuel_in_range():
    graph = create_grid(10, 10)
    assert all(0.0 <= graph.nodes[n][FUEL] <= 1.0 for n in graph.nodes)


def test_create_grid_slope_in_range():
    graph = create_grid(10, 10)
    assert all(0.0 <= graph.nodes[n][SLOPE] <= 1.0 for n in graph.nodes)


def test_create_grid_elevation_in_range():
    graph = create_grid(10, 10)
    assert all(0.0 <= graph.nodes[n][ELEVATION] <= 1.0 for n in graph.nodes)


def test_create_grid_reproducible():
    g1 = create_grid(5, 5, seed=42)
    g2 = create_grid(5, 5, seed=42)
    for node in g1.nodes:
        assert g1.nodes[node][FUEL] == g2.nodes[node][FUEL]
        assert g1.nodes[node][SLOPE] == g2.nodes[node][SLOPE]


def test_create_grid_different_seeds_differ():
    g1 = create_grid(5, 5, seed=0)
    g2 = create_grid(5, 5, seed=1)
    assert [g1.nodes[n][FUEL] for n in g1.nodes] != [g2.nodes[n][FUEL] for n in g2.nodes]


def test_create_grid_water_patches():
    graph = create_grid(20, 20, water_fraction=0.2, seed=0)
    assert sum(1 for n in graph.nodes if graph.nodes[n][FUEL] == 0.0) > 1


def test_create_grid_rock_patches():
    graph = create_grid(20, 20, rock_fraction=0.2, seed=0)
    assert sum(1 for n in graph.nodes if graph.nodes[n][FUEL] == 0.0) > 1


def test_create_grid_cell_size_default():
    graph = create_grid(3, 3)
    assert graph.graph[CELL_SIZE] == 100.0


def test_create_grid_cell_size_custom():
    graph = create_grid(3, 3, cell_size_m=30.0)
    assert graph.graph[CELL_SIZE] == 30.0


def test_create_grid_no_wind_or_moisture_set():
    graph = create_grid(3, 3)
    assert WIND_SPEED not in graph.graph
    assert WIND_DIRECTION not in graph.graph
    assert FUEL_MOISTURE not in graph.graph


# ---------------------------------------------------------------------------
# set_wind / set_fuel_moisture
# ---------------------------------------------------------------------------


def test_set_wind():
    graph = create_grid(3, 3)
    set_wind(graph, speed=30.0, direction=180.0)
    assert graph.graph[WIND_SPEED] == 30.0
    assert graph.graph[WIND_DIRECTION] == 180.0


def test_set_fuel_moisture():
    graph = create_grid(3, 3)
    set_fuel_moisture(graph, 0.25)
    assert graph.graph[FUEL_MOISTURE] == 0.25


# ---------------------------------------------------------------------------
# reset_states
# ---------------------------------------------------------------------------


def test_reset_states():
    graph = create_grid(3, 3)
    for node in list(graph.nodes)[:3]:
        graph.nodes[node][STATE] = NodeState.BURNING
    for node in list(graph.nodes)[3:5]:
        graph.nodes[node][STATE] = NodeState.BURNED

    reset_states(graph)

    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)
