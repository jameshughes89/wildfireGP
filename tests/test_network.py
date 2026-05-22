import numpy as np
import pytest

from wildfireGP.network import (
    GraphState,
    NodeState,
    TerrainType,
    create_grid,
    reset_states,
    select_ignition_cluster,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)


def test_create_grid_4by5_has_20_cells():
    state = create_grid(4, 5)
    assert state.state.shape == (4, 5)


def test_create_grid_interior_cell_has_8_neighbours():
    assert len(create_grid(5, 5).neighbours((2, 2))) == 8


def test_create_grid_edge_cell_has_5_neighbours():
    assert len(create_grid(5, 5).neighbours((0, 2))) == 5


def test_create_grid_corner_cell_has_3_neighbours():
    assert len(create_grid(5, 5).neighbours((0, 0))) == 3


def test_create_grid_default_all_cells_unburned():
    state = create_grid(4, 5)
    assert (state.state == NodeState.UNBURNED).all()


def test_create_grid_default_fuel_in_unit_interval():
    state = create_grid(10, 10)
    assert state.fuel.min() >= 0.0 and state.fuel.max() <= 1.0


def test_create_grid_default_slope_in_unit_interval():
    state = create_grid(10, 10)
    assert state.slope.min() >= 0.0 and state.slope.max() <= 1.0


def test_create_grid_default_elevation_in_unit_interval():
    state = create_grid(10, 10)
    assert state.elevation.min() >= 0.0 and state.elevation.max() <= 1.0


def test_create_grid_same_seed_is_reproducible():
    s1, s2 = create_grid(5, 5, seed=42), create_grid(5, 5, seed=42)
    assert np.array_equal(s1.fuel, s2.fuel)
    assert np.array_equal(s1.slope, s2.slope)
    assert np.array_equal(s1.elevation, s2.elevation)


def test_create_grid_different_seeds_fuel_differs():
    s1, s2 = create_grid(5, 5, seed=0), create_grid(5, 5, seed=1)
    assert not np.array_equal(s1.fuel, s2.fuel)


def test_create_grid_default_terrain_type_is_land():
    state = create_grid(10, 10, seed=0)
    assert (state.terrain == TerrainType.LAND).all()


def test_create_grid_water_fraction_exact_count():
    rows, cols, fraction = 20, 20, 0.1
    state = create_grid(rows, cols, water_fraction=fraction, seed=0)
    assert (state.terrain == TerrainType.WATER).sum() == round(rows * cols * fraction)


def test_create_grid_rock_fraction_exact_count():
    rows, cols, fraction = 20, 20, 0.1
    state = create_grid(rows, cols, rock_fraction=fraction, seed=0)
    assert (state.terrain == TerrainType.ROCK).sum() == round(rows * cols * fraction)


def test_create_grid_custom_cell_size_stored():
    assert create_grid(3, 3, cell_size_m=30.0).cell_size_m == 30.0


def test_set_wind_stores_speed():
    state = create_grid(3, 3)
    set_wind(state, speed=30.0, direction=180.0)
    assert state.wind_speed == 30.0


def test_set_wind_stores_direction():
    state = create_grid(3, 3)
    set_wind(state, speed=30.0, direction=180.0)
    assert state.wind_direction == 180.0


def test_set_wind_rejects_negative_speed():
    with pytest.raises(ValueError, match="Wind speed must be >= 0"):
        set_wind(create_grid(3, 3), speed=-1.0, direction=180.0)


def test_set_wind_rejects_negative_direction():
    with pytest.raises(ValueError, match="0 <= direction < 360"):
        set_wind(create_grid(3, 3), speed=30.0, direction=-1.0)


def test_set_wind_rejects_direction_at_upper_bound():
    with pytest.raises(ValueError, match="0 <= direction < 360"):
        set_wind(create_grid(3, 3), speed=30.0, direction=360.0)


def test_set_fuel_moisture_stores_value():
    state = create_grid(3, 3)
    set_fuel_moisture(state, 0.25)
    assert state.fuel_moisture == 0.25


def test_set_fuel_moisture_rejects_below_zero():
    with pytest.raises(ValueError, match="0 <= moisture <= 1"):
        set_fuel_moisture(create_grid(3, 3), -0.1)


def test_set_fuel_moisture_rejects_above_one():
    with pytest.raises(ValueError, match="0 <= moisture <= 1"):
        set_fuel_moisture(create_grid(3, 3), 1.1)


def test_create_grid_default_burn_timer_is_zero():
    state = create_grid(4, 5)
    assert (state.burn_timer == 0).all()


def test_reset_states_all_cells_become_unburned():
    state = create_grid(3, 3)
    state.state[0, 0] = NodeState.BURNING
    state.state[0, 1] = NodeState.BURNED
    state.state[0, 2] = NodeState.TREATED
    reset_states(state)
    assert (state.state == NodeState.UNBURNED).all()


def test_reset_states_all_burn_timers_return_to_zero():
    state = create_grid(3, 3)
    state.burn_timer[:] = 3
    reset_states(state)
    assert (state.burn_timer == 0).all()


def test_select_ignition_node_is_burnable():
    state = create_grid(20, 20, seed=0)
    node = select_ignition_node(state, np.random.default_rng(0))
    assert state.terrain[node] == TerrainType.LAND
    assert state.fuel[node] > 0.0
    assert state.state[node] == NodeState.UNBURNED


def test_select_ignition_node_is_in_central_region():
    rows, cols = 20, 20
    state = create_grid(rows, cols, seed=0)
    centre_fraction = 0.5
    margin = (1.0 - centre_fraction) / 2.0
    row_lo, row_hi = int(rows * margin), int(rows * (1.0 - margin))
    col_lo, col_hi = int(cols * margin), int(cols * (1.0 - margin))
    for _ in range(20):
        r, c = select_ignition_node(state, np.random.default_rng(_), centre_fraction=centre_fraction)
        assert row_lo <= r < row_hi
        assert col_lo <= c < col_hi


def test_select_ignition_node_is_reproducible_with_same_seed():
    state = create_grid(20, 20, seed=0)
    n1 = select_ignition_node(state, np.random.default_rng(42))
    n2 = select_ignition_node(state, np.random.default_rng(42))
    assert n1 == n2


def test_select_ignition_node_falls_back_when_centre_has_no_burnable_cells():
    state = create_grid(10, 10, seed=0)
    rows, cols = 10, 10
    margin = (1.0 - 0.5) / 2.0
    r_lo, r_hi = int(rows * margin), int(rows * (1.0 - margin))
    c_lo, c_hi = int(cols * margin), int(cols * (1.0 - margin))
    state.fuel[r_lo:r_hi, c_lo:c_hi] = 0.0
    state.terrain[r_lo:r_hi, c_lo:c_hi] = TerrainType.WATER
    node = select_ignition_node(state, np.random.default_rng(0))
    assert state.fuel[node] > 0.0
    assert state.terrain[node] == TerrainType.LAND


def test_select_ignition_node_raises_when_no_burnable_cells_exist():
    state = create_grid(5, 5, seed=0)
    state.fuel[:] = 0.0
    state.terrain[:] = TerrainType.WATER
    with pytest.raises(ValueError):
        select_ignition_node(state, np.random.default_rng(0))


def test_select_ignition_cluster_returns_requested_size():
    state = create_grid(10, 10, seed=0)
    cluster = select_ignition_cluster(state, np.random.default_rng(0), size=3)
    assert len(cluster) == 3


def test_select_ignition_cluster_all_cells_are_burnable():
    state = create_grid(10, 10, seed=0)
    cluster = select_ignition_cluster(state, np.random.default_rng(0), size=3)
    for node in cluster:
        assert state.state[node] == NodeState.UNBURNED
        assert state.terrain[node] == TerrainType.LAND
        assert state.fuel[node] > 0.0


def test_select_ignition_cluster_cells_are_unique():
    state = create_grid(10, 10, seed=0)
    cluster = select_ignition_cluster(state, np.random.default_rng(0), size=5)
    assert len(cluster) == len(set(cluster))


def test_select_ignition_cluster_size_one_equals_single_cell():
    state = create_grid(10, 10, seed=0)
    cluster = select_ignition_cluster(state, np.random.default_rng(0), size=1)
    assert len(cluster) == 1


def test_simstate_copy_is_independent():
    state = create_grid(5, 5, seed=0)
    other = state.copy()
    other.state[0, 0] = NodeState.BURNING
    other.fuel[1, 1] = 0.0
    assert state.state[0, 0] == NodeState.UNBURNED
    assert state.fuel[1, 1] != 0.0


def test_simstate_is_dataclass_instance():
    assert isinstance(create_grid(3, 3), GraphState)
