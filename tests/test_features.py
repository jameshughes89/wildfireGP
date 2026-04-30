import math

import networkx as nx
import pytest

from wildfireGP.network import (
    BURN_TIMER,
    ELEVATION,
    FUEL,
    FUEL_MOISTURE,
    SLOPE,
    STATE,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.features import (
    burn_duration,
    burned_neighbor_count,
    burning_neighbor_count,
    distance_to_fire,
    elevation,
    fuel_level,
    fuel_moisture,
    is_burned,
    is_burning,
    is_treated,
    is_unburned,
    slope,
    unburned_neighbor_count,
    wind_direction,
    wind_speed,
)
from wildfireGP.spread import MAX_BURN_STEPS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NODE = (1, 1)


def _graph():
    """3x3 land grid, all nodes UNBURNED, no wind or moisture set."""
    return create_grid(3, 3, seed=0)


def _graph_env():
    """3x3 grid with wind and moisture set."""
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


def test_burn_duration_zero_for_unburned():
    g = _graph()
    assert burn_duration(g, _NODE) == 0


def test_burn_duration_zero_for_burned():
    g = _graph()
    g.nodes[_NODE][STATE] = NodeState.BURNED
    assert burn_duration(g, _NODE) == 0


def test_burn_duration_zero_at_moment_of_ignition():
    g = _graph()
    g.nodes[_NODE][FUEL] = 0.8
    initial = max(1, math.ceil(0.8 * MAX_BURN_STEPS))
    g.nodes[_NODE][STATE] = NodeState.BURNING
    g.nodes[_NODE][BURN_TIMER] = initial
    assert burn_duration(g, _NODE) == 0


def test_burn_duration_increments_as_timer_decrements():
    g = _graph()
    g.nodes[_NODE][FUEL] = 0.8
    initial = max(1, math.ceil(0.8 * MAX_BURN_STEPS))
    g.nodes[_NODE][STATE] = NodeState.BURNING
    g.nodes[_NODE][BURN_TIMER] = initial - 2
    assert burn_duration(g, _NODE) == 2


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------

def test_burning_neighbor_count_zero_with_no_fire():
    g = _graph()
    assert burning_neighbor_count(g, _NODE) == 0


def test_burning_neighbor_count_counts_burning_neighbors():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    g.nodes[(1, 0)][STATE] = NodeState.BURNING
    assert burning_neighbor_count(g, _NODE) == 2


def test_burned_neighbor_count_counts_burned_neighbors():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNED
    assert burned_neighbor_count(g, _NODE) == 1


def test_unburned_neighbor_count_counts_unburned_neighbors():
    g = _graph()
    # centre node (1,1) on a 3x3 grid has 4 neighbours, all UNBURNED by default
    assert unburned_neighbor_count(g, _NODE) == 4


def test_unburned_neighbor_count_excludes_burning():
    g = _graph()
    g.nodes[(0, 1)][STATE] = NodeState.BURNING
    assert unburned_neighbor_count(g, _NODE) == 3


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
    # Manhattan distance from (0,0) to (4,4) = 4 + 4 = 8
    assert distance_to_fire(g, (0, 0)) == 8


def test_distance_to_fire_nearest_when_multiple_burning():
    g = create_grid(5, 5, seed=0)
    g.nodes[(0, 4)][STATE] = NodeState.BURNING   # distance from (0,0): 4
    g.nodes[(2, 2)][STATE] = NodeState.BURNING   # distance from (0,0): 4
    g.nodes[(0, 1)][STATE] = NodeState.BURNING   # distance from (0,0): 1
    assert distance_to_fire(g, (0, 0)) == 1


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
