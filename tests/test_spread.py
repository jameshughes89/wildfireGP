import math

import numpy as np
import pytest

from wildfireGP.network import (
    BURN_TIMER,
    ELEVATION,
    FUEL,
    STATE,
    TERRAIN,
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.spread import MAX_BURN_STEPS, ignition_probability, spread_step


def _setup(rows=5, cols=5, seed=0, wind_speed=20.0, wind_dir=0.0, moisture=0.2):
    graph = create_grid(rows, cols, seed=seed)
    set_wind(graph, speed=wind_speed, direction=wind_dir)
    set_fuel_moisture(graph, moisture)
    return graph


def test_spread_step_missing_wind_raises():
    graph = create_grid(3, 3)
    set_fuel_moisture(graph, 0.2)
    with pytest.raises(KeyError):
        spread_step(graph, np.random.default_rng(0))


def test_spread_step_missing_moisture_raises():
    graph = create_grid(3, 3)
    set_wind(graph, speed=10.0, direction=0.0)
    with pytest.raises(KeyError):
        spread_step(graph, np.random.default_rng(0))


def test_spread_step_burning_node_with_timer_one_transitions_to_burned():
    graph = _setup()
    node = (2, 2)
    graph.nodes[node][STATE] = NodeState.BURNING
    graph.nodes[node][BURN_TIMER] = 1
    spread_step(graph, np.random.default_rng(0))
    assert graph.nodes[node][STATE] == NodeState.BURNED


def test_spread_step_burning_node_decrements_timer():
    graph = _setup()
    node = (2, 2)
    graph.nodes[node][STATE] = NodeState.BURNING
    graph.nodes[node][BURN_TIMER] = 3
    spread_step(graph, np.random.default_rng(0))
    assert graph.nodes[node][BURN_TIMER] == 2
    assert graph.nodes[node][STATE] == NodeState.BURNING


def test_spread_step_ignited_node_burn_timer_set_from_fuel():
    graph = _setup(moisture=0.0, wind_speed=0.0)
    src = (2, 2)
    dst = (2, 3)
    graph.nodes[src][STATE] = NodeState.BURNING
    graph.nodes[src][BURN_TIMER] = 1
    graph.nodes[dst][FUEL] = 0.8
    graph.nodes[dst][TERRAIN] = TerrainType.LAND
    spread_step(graph, np.random.default_rng(0))
    if graph.nodes[dst][STATE] == NodeState.BURNING:
        expected = max(1, math.ceil(0.8 * MAX_BURN_STEPS))
        assert graph.nodes[dst][BURN_TIMER] == expected


def test_spread_step_no_burning_nodes_leaves_graph_unchanged():
    graph = _setup()
    states_before = [graph.nodes[n][STATE] for n in graph.nodes]
    spread_step(graph, np.random.default_rng(0))
    assert [graph.nodes[n][STATE] for n in graph.nodes] == states_before


def test_spread_step_treated_node_never_ignites():
    graph = _setup(seed=1, moisture=0.0)
    centre = (2, 2)
    graph.nodes[centre][STATE] = NodeState.BURNING
    for nb in graph.neighbors(centre):
        graph.nodes[nb][STATE] = NodeState.TREATED
    spread_step(graph, np.random.default_rng(0))
    assert all(graph.nodes[nb][STATE] == NodeState.TREATED for nb in graph.neighbors(centre))


def test_spread_step_water_node_never_ignites():
    graph = _setup(seed=1, moisture=0.0)
    centre = (2, 2)
    graph.nodes[centre][STATE] = NodeState.BURNING
    for nb in graph.neighbors(centre):
        graph.nodes[nb][TERRAIN] = TerrainType.WATER
    spread_step(graph, np.random.default_rng(0))
    assert all(graph.nodes[nb][STATE] == NodeState.UNBURNED for nb in graph.neighbors(centre))


def test_spread_step_rock_node_never_ignites():
    graph = _setup(seed=1, moisture=0.0)
    centre = (2, 2)
    graph.nodes[centre][STATE] = NodeState.BURNING
    for nb in graph.neighbors(centre):
        graph.nodes[nb][TERRAIN] = TerrainType.ROCK
    spread_step(graph, np.random.default_rng(0))
    assert all(graph.nodes[nb][STATE] == NodeState.UNBURNED for nb in graph.neighbors(centre))


def test_ignition_probability_saturated_moisture_is_zero():
    graph = _setup(moisture=1.0)
    src, dst = (2, 2), (2, 3)
    assert ignition_probability(graph, src, dst) == 0.0


def test_ignition_probability_water_terrain_is_zero():
    graph = _setup(moisture=0.0)
    src, dst = (2, 2), (2, 3)
    graph.nodes[dst][TERRAIN] = TerrainType.WATER
    assert ignition_probability(graph, src, dst) == 0.0


def test_ignition_probability_rock_terrain_is_zero():
    graph = _setup(moisture=0.0)
    src, dst = (2, 2), (2, 3)
    graph.nodes[dst][TERRAIN] = TerrainType.ROCK
    assert ignition_probability(graph, src, dst) == 0.0


def test_ignition_probability_downwind_higher_than_upwind():
    graph = _setup(moisture=0.1, wind_speed=30.0, wind_dir=0.0)
    src = (2, 2)
    north = (1, 2)
    south = (3, 2)
    assert ignition_probability(graph, src, south) > ignition_probability(graph, src, north)


def test_ignition_probability_uphill_higher_than_downhill():
    graph = _setup(moisture=0.1, wind_speed=0.0, wind_dir=0.0)
    src = (2, 2)
    uphill = (2, 3)
    downhill = (2, 1)
    graph.nodes[uphill][TERRAIN] = TerrainType.LAND
    graph.nodes[downhill][TERRAIN] = TerrainType.LAND
    graph.nodes[uphill][FUEL] = 0.8
    graph.nodes[downhill][FUEL] = 0.8
    graph.nodes[src][ELEVATION] = 0.5
    graph.nodes[uphill][ELEVATION] = 0.6
    graph.nodes[downhill][ELEVATION] = 0.4
    assert ignition_probability(graph, src, uphill) > ignition_probability(graph, src, downhill)


def test_ignition_probability_result_in_unit_interval():
    graph = _setup(moisture=0.0, wind_speed=50.0, wind_dir=45.0)
    src, dst = (2, 2), (2, 3)
    p = ignition_probability(graph, src, dst)
    assert 0.0 <= p <= 1.0
