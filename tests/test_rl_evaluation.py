"""Tests for the canonical-harness evaluation adapter.

as_scoring_function caches a whole-grid scoring pass and serves it per cell. If that cache ever goes stale, every
evaluated number is silently wrong in a way no downstream check would catch --- the scores would simply be those of an
earlier timestep. These tests pin the invalidation, using a fixed random linear scorer as the learned policy.
"""

import numpy as np
import pytest

from wildfireGP.evaluate import evaluate, init_ignition, simulate
from wildfireGP.features import update_neighbourhood_maps_after_treatment
from wildfireGP.network import (
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.rl.evaluation import as_scoring_function
from wildfireGP.rl.features import (
    N_FEATURES,
    compute_action_mask,
    extract_features,
    run_precomputes,
)
from wildfireGP.spread import spread_step


def make_scorer():
    weights = np.random.default_rng(123).standard_normal(N_FEATURES)

    def scorer(features):
        return features @ weights

    return scorer


def make_state(seed=0, rows=30, cols=30):
    state = create_grid(rows=rows, cols=cols, seed=seed)
    set_wind(state, speed=36.0, direction=140.0)
    set_fuel_moisture(state, moisture=0.2)
    land = np.argwhere((state.terrain == TerrainType.LAND) & (state.fuel > 0.3))
    r, c = land[len(land) // 2]
    init_ignition(state, [(int(r), int(c))])
    return state


def test_scores_match_a_direct_batched_forward_pass():
    scorer = make_scorer()
    scoring = as_scoring_function(scorer)
    state = make_state()
    rng = np.random.default_rng(0)
    for _ in range(6):
        spread_step(state, rng)
    run_precomputes(state)

    expected = scorer(extract_features(state))
    for flat in range(0, state.rows * state.cols, 37):
        node = divmod(flat, state.cols)
        assert scoring(state, node) == pytest.approx(expected[flat], abs=1e-6)


def test_cache_invalidates_when_a_treatment_is_placed():
    """_apply_treatments rescores neighbours after each placement; a stale cache would return pre-treatment scores."""
    scorer = make_scorer()
    scoring = as_scoring_function(scorer)
    state = make_state()
    rng = np.random.default_rng(1)
    for _ in range(6):
        spread_step(state, rng)
    run_precomputes(state)

    eligible = np.flatnonzero(compute_action_mask(state))
    target = int(eligible[len(eligible) // 2])
    treated = int(eligible[0])
    scoring(state, divmod(target, state.cols))  # populate the cache for this timestep

    node = divmod(treated, state.cols)
    state.state[node] = NodeState.TREATED
    state.step_treated += 1
    state.step_unburned -= 1
    update_neighbourhood_maps_after_treatment(state, node)

    fresh = scorer(extract_features(state))
    after = scoring(state, divmod(target, state.cols))
    assert after == pytest.approx(fresh[target], abs=1e-6)
    # The treated cell itself must now be scored from its post-treatment features, not the cached pre-treatment ones.
    assert scoring(state, node) == pytest.approx(fresh[treated], abs=1e-6)


def test_cache_invalidates_across_timesteps():
    scorer = make_scorer()
    scoring = as_scoring_function(scorer)
    state = make_state()
    rng = np.random.default_rng(2)
    seen = []
    for _ in range(8):
        spread_step(state, rng)
        run_precomputes(state)
        fresh = scorer(extract_features(state))
        probes = [int(f) for f in np.linspace(0, state.rows * state.cols - 1, 25).astype(int)]
        for flat in probes:
            assert scoring(state, divmod(flat, state.cols)) == pytest.approx(fresh[flat], abs=1e-6)
        seen.append(fresh[probes[12]])
    assert len(set(np.round(seen, 6))) > 1, "scores should change as the fire evolves"


def test_stays_fresh_through_a_full_simulation():
    """End-to-end: drive simulate() with the adapter and confirm every served score matches a fresh computation."""
    scorer = make_scorer()
    scoring = as_scoring_function(scorer)
    state = make_state()
    rng = np.random.default_rng(3)
    mismatches = 0
    for _step, live in simulate(state, scoring, 3, 30, rng, 3):
        run_precomputes(live)
        fresh = scorer(extract_features(live))
        for flat in (0, 211, 457, 700, 899):
            if abs(scoring(live, divmod(flat, live.cols)) - fresh[flat]) > 1e-5:
                mismatches += 1
    assert mismatches == 0


def test_annotated_with_the_precomputes_the_features_need():
    """simulate() reads this annotation to decide which precomputes to run; a wrong set means wrong features."""
    scoring = as_scoring_function(make_scorer())
    required = scoring._required_precomputes
    assert {
        "fire_map",
        "burnable_fire_map",
        "neighbourhood_maps",
        "state_counts",
        "reachable_unburned_area",
        "burning_two_hop_map",
    } <= set(required)
    assert not {"betweenness_static", "betweenness_dynamic", "mtt_pathway_count"} & set(
        required
    ), "graph-theoretic precomputes are absent from the GP language and ~100x slower"


def test_runs_through_the_canonical_harness():
    scorer = make_scorer()
    state = make_state()
    template = create_grid(rows=30, cols=30, seed=0)
    set_wind(template, speed=36.0, direction=140.0)
    set_fuel_moisture(template, moisture=0.2)
    ignition = [(int(r), int(c)) for r, c in np.argwhere(state.state == NodeState.BURNING)]

    burned, peak = evaluate(as_scoring_function(scorer), template, ignition, 3, 40, np.random.default_rng(0), 3, 0)
    assert 0 < burned <= 900
    assert peak > 0
