import math

import numpy as np

from wildfireGP.features import precompute_fire_map, wind_fire_alignment
from wildfireGP.network import (
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.spread import MAX_BURN_STEPS, ignition_probability, spread_step


def _setup(rows=5, cols=5, seed=0, wind_speed=20.0, wind_dir=0.0, moisture=0.2):
    state = create_grid(rows, cols, seed=seed)
    set_wind(state, speed=wind_speed, direction=wind_dir)
    set_fuel_moisture(state, moisture)
    return state


def test_spread_step_burning_cell_with_timer_one_transitions_to_burned():
    state = _setup()
    node = (2, 2)
    state.state[node] = NodeState.BURNING
    state.burn_timer[node] = 1
    spread_step(state, np.random.default_rng(0))
    assert state.state[node] == NodeState.BURNED


def test_spread_step_burned_out_cell_fuel_zeroed():
    state = _setup()
    node = (2, 2)
    state.state[node] = NodeState.BURNING
    state.burn_timer[node] = 1
    state.fuel[node] = 0.9
    spread_step(state, np.random.default_rng(0))
    assert state.state[node] == NodeState.BURNED
    assert state.fuel[node] == 0.0


def test_spread_step_burning_cell_decrements_timer():
    state = _setup()
    node = (2, 2)
    state.state[node] = NodeState.BURNING
    state.burn_timer[node] = 3
    spread_step(state, np.random.default_rng(0))
    assert state.burn_timer[node] == 2
    assert state.state[node] == NodeState.BURNING


def test_spread_step_ignited_cell_burn_timer_set_from_fuel():
    state = _setup(moisture=0.0, wind_speed=0.0)
    src = (2, 2)
    dst = (2, 3)
    state.state[src] = NodeState.BURNING
    state.burn_timer[src] = 1
    state.fuel[dst] = 0.8
    state.terrain[dst] = TerrainType.LAND
    spread_step(state, np.random.default_rng(0))
    if state.state[dst] == NodeState.BURNING:
        expected = max(1, math.ceil(0.8 * MAX_BURN_STEPS))
        assert state.burn_timer[dst] == expected


def test_spread_step_no_burning_cells_leaves_state_unchanged():
    state = _setup()
    states_before = state.state.copy()
    spread_step(state, np.random.default_rng(0))
    assert np.array_equal(state.state, states_before)


def test_spread_step_treated_cell_never_ignites():
    state = _setup(seed=1, moisture=0.0)
    centre = (2, 2)
    state.state[centre] = NodeState.BURNING
    for nb in state.neighbours(centre):
        state.state[nb] = NodeState.TREATED
    spread_step(state, np.random.default_rng(0))
    assert all(state.state[nb] == NodeState.TREATED for nb in state.neighbours(centre))


def test_spread_step_water_cell_never_ignites():
    state = _setup(seed=1, moisture=0.0)
    centre = (2, 2)
    state.state[centre] = NodeState.BURNING
    for nb in state.neighbours(centre):
        state.terrain[nb] = TerrainType.WATER
    spread_step(state, np.random.default_rng(0))
    assert all(state.state[nb] == NodeState.UNBURNED for nb in state.neighbours(centre))


def test_spread_step_rock_cell_never_ignites():
    state = _setup(seed=1, moisture=0.0)
    centre = (2, 2)
    state.state[centre] = NodeState.BURNING
    for nb in state.neighbours(centre):
        state.terrain[nb] = TerrainType.ROCK
    spread_step(state, np.random.default_rng(0))
    assert all(state.state[nb] == NodeState.UNBURNED for nb in state.neighbours(centre))


def test_ignition_probability_saturated_moisture_is_zero():
    state = _setup(moisture=1.0)
    src, dst = (2, 2), (2, 3)
    assert ignition_probability(state, src, dst) == 0.0


def test_ignition_probability_water_terrain_is_zero():
    state = _setup(moisture=0.0)
    src, dst = (2, 2), (2, 3)
    state.terrain[dst] = TerrainType.WATER
    assert ignition_probability(state, src, dst) == 0.0


def test_ignition_probability_rock_terrain_is_zero():
    state = _setup(moisture=0.0)
    src, dst = (2, 2), (2, 3)
    state.terrain[dst] = TerrainType.ROCK
    assert ignition_probability(state, src, dst) == 0.0


def test_ignition_probability_downwind_higher_than_upwind():
    state = _setup(moisture=0.1, wind_speed=30.0, wind_dir=0.0)
    src = (2, 2)
    north = (1, 2)
    south = (3, 2)
    assert ignition_probability(state, src, south) > ignition_probability(state, src, north)


def test_wind_from_direction_convention_matches_feature_and_spread():
    state = _setup(moisture=0.1, wind_speed=30.0, wind_dir=0.0)
    src = (2, 2)
    north = (1, 2)
    south = (3, 2)

    state.state[src] = NodeState.BURNING
    precompute_fire_map(state)

    assert wind_fire_alignment(state, south) > wind_fire_alignment(state, north)
    assert ignition_probability(state, src, south) > ignition_probability(state, src, north)


def test_ignition_probability_uphill_higher_than_downhill():
    state = _setup(moisture=0.1, wind_speed=0.0, wind_dir=0.0)
    src = (2, 2)
    uphill = (2, 3)
    downhill = (2, 1)
    state.terrain[uphill] = TerrainType.LAND
    state.terrain[downhill] = TerrainType.LAND
    state.fuel[uphill] = 0.8
    state.fuel[downhill] = 0.8
    state.elevation[src] = 0.5
    state.elevation[uphill] = 0.6
    state.elevation[downhill] = 0.4
    assert ignition_probability(state, src, uphill) > ignition_probability(state, src, downhill)


def test_ignition_probability_slope_effect_is_non_trivial():
    state = _setup(moisture=0.1, wind_speed=0.0, wind_dir=0.0)
    src = (2, 2)
    uphill = (2, 3)
    flat_nbr = (2, 1)
    for node in (uphill, flat_nbr):
        state.terrain[node] = TerrainType.LAND
        state.fuel[node] = 0.8
    state.elevation[src] = 0.0
    state.elevation[uphill] = 1.0
    state.elevation[flat_nbr] = 0.0
    p_uphill = ignition_probability(state, src, uphill)
    p_flat = ignition_probability(state, src, flat_nbr)
    assert p_uphill / p_flat > 1.05


def test_ignition_probability_slope_matches_alexandridis_atan_formula():
    state = _setup(moisture=0.0, wind_speed=0.0, wind_dir=0.0)
    src = (2, 2)
    uphill = (2, 3)
    state.terrain[uphill] = TerrainType.LAND
    state.fuel[uphill] = 0.5
    state.elevation[src] = 0.0
    state.elevation[uphill] = 1.0
    expected = 0.5 * math.exp(0.078 * math.atan(1.0))
    assert math.isclose(ignition_probability(state, src, uphill), expected, rel_tol=1e-6)


def test_ignition_probability_result_in_unit_interval():
    state = _setup(moisture=0.0, wind_speed=50.0, wind_dir=45.0)
    src, dst = (2, 2), (2, 3)
    p = ignition_probability(state, src, dst)
    assert 0.0 <= p <= 1.0
