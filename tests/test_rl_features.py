"""Tests for the vectorised RL feature extractor.

The load-bearing assertion is that the vectorised (N, 22) matrix agrees cell-for-cell with the per-cell functions in
``wildfireGP.features``. Nothing downstream can detect a scrambled or stale column: the MLP would simply learn on the
wrong inputs and the head-to-head comparison would be silently invalid.
"""

import numpy as np
import pytest

from wildfireGP.evaluate import init_ignition
from wildfireGP.features import (
    burnable_distance_to_fire,
    burning_neighbour_count,
    burning_two_hop_count,
    distance_to_fire,
    elevation,
    elevation_delta_to_fire,
    fuel_level,
    fuel_moisture,
    has_treated_neighbour,
    mean_neighbour_elevation,
    mean_neighbour_fuel,
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
    wind_speed,
)
from wildfireGP.network import (
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.rl.features import (
    FEATURE_ORDER,
    N_FEATURES,
    compute_action_mask,
    default_unreachable_distance,
    extract_features,
    run_precomputes,
)
from wildfireGP.spread import spread_step

NODE_FUNCS = {
    "fuel_level": fuel_level,
    "elevation": elevation,
    "slope": slope,
    "mean_neighbour_elevation": mean_neighbour_elevation,
    "mean_neighbour_fuel": mean_neighbour_fuel,
    "burning_neighbour_count": burning_neighbour_count,
    "burning_two_hop_count": burning_two_hop_count,
    "unburned_neighbour_count": unburned_neighbour_count,
    "unburnable_neighbour_count": unburnable_neighbour_count,
    "has_treated_neighbour": has_treated_neighbour,
    "treated_neighbour_count": treated_neighbour_count,
    "distance_to_fire": distance_to_fire,
    "burnable_distance_to_fire": burnable_distance_to_fire,
    "reachable_unburned_area": reachable_unburned_area,
    "wind_fire_alignment": wind_fire_alignment,
    "elevation_delta_to_fire": elevation_delta_to_fire,
}
GRAPH_FUNCS = {
    "wind_speed": wind_speed,
    "fuel_moisture": fuel_moisture,
    "total_burning": total_burning,
    "total_burned": total_burned,
    "total_unburned": total_unburned,
    "total_treated": total_treated,
}


def reference_features(state, sentinel):
    """Build the (N, 22) matrix the slow way: one wildfireGP function call per cell per feature."""
    out = np.empty((state.rows * state.cols, N_FEATURES), dtype=np.float64)
    graph_values = [GRAPH_FUNCS[name](state) for name in FEATURE_ORDER[16:]]
    for r in range(state.rows):
        for c in range(state.cols):
            i = r * state.cols + c
            for j, name in enumerate(FEATURE_ORDER[:16]):
                out[i, j] = NODE_FUNCS[name](state, (r, c))
            out[i, 16:] = graph_values
    return np.where(np.isfinite(out), out, sentinel).astype(np.float32)


def make_state(seed, water=0.0, rock=0.0):
    state = create_grid(rows=30, cols=30, water_fraction=water, rock_fraction=rock, seed=seed)
    set_wind(state, speed=20.0, direction=135.0)
    set_fuel_moisture(state, moisture=0.2)
    land = np.argwhere((state.terrain == TerrainType.LAND) & (state.fuel > 0.3))
    r, c = land[len(land) // 2]
    init_ignition(state, [(int(r), int(c))])
    return state


def test_feature_order_matches_gp_language():
    """Drift guard: wildfireGP/ here is a mutable copy, so a feature added to the language must not go unnoticed."""
    from wildfireGP.language import _GRAPH_FEATURES, _NODE_FEATURES

    gp_order = tuple(f.__name__ for f in _NODE_FEATURES) + tuple(f.__name__ for f in _GRAPH_FEATURES)
    assert FEATURE_ORDER == gp_order
    assert N_FEATURES == 22


def test_feature_order_has_no_graph_theoretic_features():
    """betweenness_* and mtt_pathway_count are ~100x slower and absent from the GP language; the RL must not see
    them."""
    assert not {"betweenness_static", "betweenness_dynamic", "mtt_pathway_count"} & set(FEATURE_ORDER)


@pytest.mark.parametrize("water,rock", [(0.0, 0.0), (0.15, 0.10)])
@pytest.mark.parametrize("seed", [0, 7])
def test_vectorised_matches_per_cell_functions(seed, water, rock):
    """Assert agreement across the fire's lifetime, not just one frame.

    The neighbourhood and fire-distance maps are degenerate before the fire spreads, so a single early snapshot would
    pass even with several columns wrong.
    """
    state = make_state(seed, water, rock)
    rng = np.random.default_rng(seed)
    sentinel = default_unreachable_distance(state)

    for step in range(18):
        if not (state.state == NodeState.BURNING).any():
            break
        spread_step(state, rng)
        run_precomputes(state)
        np.testing.assert_allclose(extract_features(state), reference_features(state, sentinel), rtol=1e-6, atol=1e-6)


def test_vectorised_matches_after_mid_timestep_treatments():
    """Treatments mutate the neighbourhood maps incrementally; extraction must track that, not fresh precomputes."""
    state = make_state(3)
    rng = np.random.default_rng(3)
    for _ in range(8):
        spread_step(state, rng)
    run_precomputes(state)

    sentinel = default_unreachable_distance(state)
    eligible = np.flatnonzero(compute_action_mask(state))
    for flat in eligible[:: len(eligible) // 3][:3]:
        node = divmod(int(flat), state.cols)
        state.state[node] = NodeState.TREATED
        state.step_treated += 1
        state.step_unburned -= 1
        update_neighbourhood_maps_after_treatment(state, node)
        np.testing.assert_allclose(extract_features(state), reference_features(state, sentinel), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("water,rock", [(0.0, 0.0), (0.15, 0.10)])
def test_features_are_always_finite(water, rock):
    """An inf reaching the first Linear NaNs the whole forward pass; barrier landscapes make this common."""
    state = make_state(1, water, rock)
    rng = np.random.default_rng(1)
    run_precomputes(state)
    for _ in range(25):
        assert np.isfinite(extract_features(state)).all()
        if not (state.state == NodeState.BURNING).any():
            break
        spread_step(state, rng)
        run_precomputes(state)
    assert np.isfinite(extract_features(state)).all()


def test_unreachable_distance_is_beyond_every_reachable_cell():
    state = make_state(1, water=0.15, rock=0.10)
    rng = np.random.default_rng(1)
    for _ in range(10):
        spread_step(state, rng)
    run_precomputes(state)

    sentinel = default_unreachable_distance(state)
    features = extract_features(state)
    burnable = np.array(
        [burnable_distance_to_fire(state, (r, c)) for r in range(state.rows) for c in range(state.cols)]
    )
    unreachable = ~np.isfinite(burnable)
    assert unreachable.any(), "expected some unreachable cells on a barrier landscape"
    assert (features[unreachable, 12] == sentinel).all()
    assert features[~unreachable, 12].max() < sentinel


def test_extract_requires_precomputes():
    state = make_state(0)
    with pytest.raises(RuntimeError, match="run_precomputes"):
        extract_features(state)


def test_no_fire_present_yields_sentinel_distances():
    state = create_grid(rows=20, cols=20, seed=0)
    set_wind(state, speed=20.0, direction=0.0)
    set_fuel_moisture(state, moisture=0.2)
    run_precomputes(state)
    features = extract_features(state)
    sentinel = default_unreachable_distance(state)
    assert np.isfinite(features).all()
    assert (features[:, 11] == sentinel).all()
    assert (features[:, 12] == sentinel).all()


def test_action_mask_matches_gp_candidate_pool():
    """The mask must equal the candidate set wildfireGP.evaluate._apply_treatments would build."""
    state = make_state(5)
    rng = np.random.default_rng(5)
    for _ in range(10):
        spread_step(state, rng)
    run_precomputes(state)

    expected = (state.state == NodeState.UNBURNED) & (state.fuel > 0.0)
    np.testing.assert_array_equal(compute_action_mask(state), expected.reshape(-1))

    gated = compute_action_mask(state, min_treatment_distance=3)
    np.testing.assert_array_equal(gated, (expected & (state.fire_distance_map >= 3)).reshape(-1))
    assert gated.sum() < expected.sum()
