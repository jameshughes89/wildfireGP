import networkx as nx

from wildfireGP.network import (
    BURN_TIMER,
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
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.features import (
    burn_steps_remaining,
    burning_neighbour_count,
    distance_to_fire,
    elevation,
    fuel_level,
    fuel_moisture,
    is_burned,
    is_burning,
    is_treated,
    is_unburned,
    slope,
    total_burned,
    total_burning,
    total_treated,
    total_unburned,
    unburnable_neighbour_count,
    unburned_neighbour_count,
    wind_direction,
    wind_fire_alignment,
    wind_speed,
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

def test_fuel_level_returns_node_fuel():
    g = _graph()
    assert fuel_level(g, _NODE) == g.nodes[_NODE][FUEL]


def test_fuel_level_is_zero_when_set_to_zero():
    g = _graph()
    g.nodes[_NODE][FUEL] = 0.0
    assert fuel_level(g, _NODE) == 0.0


def test_elevation_returns_node_elevation():
    g = _graph()
    assert elevation(g, _NODE) == g.nodes[_NODE][ELEVATION]


def test_slope_returns_node_slope():
    g = _graph()
    assert slope(g, _NODE) == g.nodes[_NODE][SLOPE]


# ---------------------------------------------------------------------------
# Fire state
# ---------------------------------------------------------------------------

def test_is_unburned_true_by_default():
    g = _graph()
    assert is_unburned(g, _NODE)


def test_is_burning_true_when_burning():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    assert is_burning(g, _NODE)


def test_is_burning_false_when_unburned():
    g = _graph()
    assert not is_burning(g, _NODE)


def test_is_burned_true_when_burned():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNED
    assert is_burned(g, _NODE)


def test_is_treated_true_when_treated():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.TREATED
    assert is_treated(g, _NODE)


def test_burn_steps_remaining_zero_for_unburned():
    g = _graph()
    assert burn_steps_remaining(g, _NODE) == 0


def test_burn_steps_remaining_returns_burn_timer():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    g.nodes[_NODE][BURN_TIMER] = 3
    assert burn_steps_remaining(g, _NODE) == 3


def test_burn_steps_remaining_decreases_as_fire_progresses():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    g.nodes[_NODE][BURN_TIMER] = 4
    assert burn_steps_remaining(g, _NODE) > 0
    g.nodes[_NODE][BURN_TIMER] = 1
    assert burn_steps_remaining(g, _NODE) == 1


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------

def test_burning_neighbour_count_zero_with_no_fire():
    g = _graph()
    assert burning_neighbour_count(g, _NODE) == 0


def test_burning_neighbour_count_counts_burning_neighbours():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    g.nodes[(1, 0)][STATE] = NodeState.BURNING
    assert burning_neighbour_count(g, _NODE) == 2


def test_unburned_neighbour_count_all_unburned_by_default():
    g = _graph()
    assert unburned_neighbour_count(g, _NODE) == 4


def test_unburned_neighbour_count_excludes_burning():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    assert unburned_neighbour_count(g, _NODE) == 3


def test_unburnable_neighbour_count_includes_burned():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNED
    assert unburnable_neighbour_count(g, _NODE) == 1


def test_unburnable_neighbour_count_includes_treated():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.TREATED
    assert unburnable_neighbour_count(g, _NODE) == 1


def test_unburnable_neighbour_count_includes_water():
    g = _graph()
    g.nodes[(0, 1)][TERRAIN] = TerrainType.WATER
    assert unburnable_neighbour_count(g, _NODE) == 1


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


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

def test_distance_to_fire_returns_inf_with_no_fire():
    g = _graph()
    assert distance_to_fire(g, _NODE) == float("inf")


def test_distance_to_fire_zero_when_node_is_burning():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    assert distance_to_fire(g, _NODE) == 0


def test_distance_to_fire_manhattan_distance():
    g = create_grid(5, 5, seed=0)
    g.nodes[(4, 4)][STATE] = NodeState.BURNING
    assert distance_to_fire(g, (0, 0)) == 8


def test_distance_to_fire_nearest_when_multiple_burning():
    g = create_grid(5, 5, seed=0)
    g.nodes[(0, 4)][STATE] = NodeState.BURNING
    g.nodes[(2, 2)][STATE] = NodeState.BURNING
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    assert distance_to_fire(g, (0, 0)) == 1


def test_wind_fire_alignment_zero_with_no_fire():
    g = _graph_env()
    assert wind_fire_alignment(g, _NODE) == 0.0


def test_wind_fire_alignment_zero_when_node_is_burning():
    g = _graph_env()
    g.nodes[_NODE][STATE] = NodeState.BURNING
    assert wind_fire_alignment(g, _NODE) == 0.0


def test_wind_fire_alignment_positive_one_when_directly_downwind():
    # Wind from north (direction=0, blowing south). Fire directly north of node.
    # Node at (4,2), fire at (0,2) — fire is north, wind pushes it south toward node.
    g = create_grid(5, 5, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    g.nodes[(0, 2)][STATE] = NodeState.BURNING
    assert abs(wind_fire_alignment(g, (4, 2)) - 1.0) < 1e-9


def test_wind_fire_alignment_negative_one_when_directly_upwind():
    # Wind from north (direction=0, blowing south). Fire directly south of node.
    # Node at (0,2), fire at (4,2) — fire is south, wind pushes it away from node.
    g = create_grid(5, 5, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    g.nodes[(4, 2)][STATE] = NodeState.BURNING
    assert abs(wind_fire_alignment(g, (0, 2)) - (-1.0)) < 1e-9


def test_wind_fire_alignment_zero_when_crosswind():
    # Wind from north (direction=0, blowing south). Fire directly east of node.
    # No north/south component to the fire-to-node vector.
    g = create_grid(5, 5, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    g.nodes[(2, 4)][STATE] = NodeState.BURNING
    assert abs(wind_fire_alignment(g, (2, 0))) < 1e-9


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def test_wind_speed_returns_graph_attribute():
    g = _graph_env()
    assert wind_speed(g) == 15.0


def test_wind_direction_returns_graph_attribute():
    g = _graph_env()
    assert wind_direction(g) == 90.0


def test_fuel_moisture_returns_graph_attribute():
    g = _graph_env()
    assert fuel_moisture(g) == 0.2


# ---------------------------------------------------------------------------
# Whole-graph state
# ---------------------------------------------------------------------------

def test_total_burning_counts_burning_nodes():
    g = _graph()
    g.nodes[(0, 0)][STATE] = NodeState.BURNING
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    assert total_burning(g) == 2


def test_total_burned_counts_burned_nodes():
    g = _graph()
    g.nodes[(0, 0)][STATE] = NodeState.BURNED
    assert total_burned(g) == 1


def test_total_unburned_counts_unburned_nodes():
    g = _graph()
    assert total_unburned(g) == 9


def test_total_treated_counts_treated_nodes():
    g = _graph()
    g.nodes[(0, 0)][STATE] = NodeState.TREATED
    g.nodes[(1, 1)][STATE] = NodeState.TREATED
    assert total_treated(g) == 2


def test_totals_sum_to_node_count():
    g = _graph()
    g.nodes[(0, 0)][STATE] = NodeState.BURNING
    g.nodes[(0, 1)][STATE] = NodeState.BURNED
    g.nodes[(0, 2)][STATE] = NodeState.TREATED
    total = total_burning(g) + total_burned(g) + total_unburned(g) + total_treated(g)
    assert total == len(g.nodes)
