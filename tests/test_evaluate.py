import numpy as np

from wildfireGP.evaluate import (
    _apply_treatments,
    _safe_score,
    annotate_required_precomputes,
    evaluate,
    init_ignition,
    simulate,
)
from wildfireGP.features import (
    distance_to_fire,
    precompute_fire_map,
    precompute_state_counts,
    treated_neighbour_count,
)
from wildfireGP.network import (
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)


def _setup(rows=7, cols=7, moisture=0.1, wind_speed=10.0, wind_dir=0.0, seed=0):
    s = create_grid(rows, cols, seed=seed)
    set_wind(s, speed=wind_speed, direction=wind_dir)
    set_fuel_moisture(s, moisture=moisture)
    return s


def _no_op(state, node):
    return 0.0


def _rng():
    return np.random.default_rng(0)


def test_evaluate_returns_two_ints():
    s = _setup()
    result = evaluate(_no_op, s, [(3, 3)], treatments_per_step=1, max_steps=5, rng=_rng())
    assert len(result) == 2
    assert all(isinstance(v, int) for v in result)


def test_evaluate_no_ignition_returns_zero_burned():
    s = _setup()
    total_burned, peak_burning = evaluate(_no_op, s, [], treatments_per_step=1, max_steps=10, rng=_rng())
    assert total_burned == 0
    assert peak_burning == 0


def test_evaluate_saturated_moisture_fire_does_not_spread():
    s = _setup(moisture=1.0)
    ignition = (3, 3)
    total_burned, _ = evaluate(_no_op, s, [ignition], treatments_per_step=0, max_steps=10, rng=_rng())
    assert total_burned <= 1


def test_evaluate_does_not_modify_original_state():
    s = _setup()
    state_before = s.state.copy()
    evaluate(_no_op, s, [(3, 3)], treatments_per_step=1, max_steps=5, rng=_rng())
    assert np.array_equal(s.state, state_before)


def test_evaluate_peak_burning_at_least_ignition_count():
    s = _setup()
    ignitions = [(3, 3), (3, 4)]
    _, peak_burning = evaluate(_no_op, s, ignitions, treatments_per_step=0, max_steps=10, rng=_rng())
    assert peak_burning >= len(ignitions)


def test_evaluate_peak_burning_not_greater_than_cell_count():
    s = _setup()
    _, peak_burning = evaluate(_no_op, s, [(3, 3)], treatments_per_step=0, max_steps=20, rng=_rng())
    assert peak_burning <= s.rows * s.cols


def test_evaluate_peak_burning_counts_ignitions_that_burn_out_without_spreading():
    s = _setup(moisture=1.0)
    ignitions = [(3, 3), (3, 4)]
    for n in ignitions:
        s.fuel[n] = 0.2
    _, peak_burning = evaluate(_no_op, s, ignitions, treatments_per_step=0, max_steps=10, rng=_rng())
    assert peak_burning >= len(ignitions)


def test_evaluate_zero_treatments_allows_spread():
    s = _setup(moisture=0.05)
    total_burned, _ = evaluate(_no_op, s, [(3, 3)], treatments_per_step=0, max_steps=20, rng=_rng())
    assert total_burned > 1


def test_evaluate_treating_nearest_cells_reduces_burned_area_smoke():
    s = _setup(moisture=0.05, seed=1)
    ignition = [(3, 3)]

    def no_treatment(state, node):
        return 0.0

    def nearest_first(state, node):
        precompute_fire_map(state)
        d = distance_to_fire(state, node)
        return -d if d != float("inf") else float("-inf")

    burned_no_treatment, _ = evaluate(no_treatment, s, ignition, treatments_per_step=3, max_steps=15, rng=_rng())
    burned_nearest, _ = evaluate(nearest_first, s, ignition, treatments_per_step=3, max_steps=15, rng=_rng())
    assert burned_nearest <= burned_no_treatment


def test_evaluate_intervention_delay_no_func_calls_before_delay():
    s = _setup(moisture=0.05, seed=0)
    call_count = [0]

    def counting(state, node):
        call_count[0] += 1
        return 1.0

    evaluate(counting, s, [(3, 3)], treatments_per_step=5, max_steps=5, rng=_rng(), intervention_delay=5)
    assert call_count[0] == 0


def test_evaluate_intervention_delay_first_treatment_at_delay_step():
    s = _setup(moisture=0.05, seed=0)
    init_ignition(s, [(3, 3)])
    first_treated_step = None
    for step, sim_state in simulate(
        s, lambda state, node: 1.0, treatments_per_step=5, max_steps=10, rng=_rng(), intervention_delay=3
    ):
        if (sim_state.state == NodeState.TREATED).any():
            first_treated_step = step
            break
    assert first_treated_step == 3


def test_evaluate_nan_score_treated_last():
    s = _setup(moisture=1.0)
    ignition = (3, 3)

    nan_count = 0

    def nan_for_all(state, node):
        nonlocal nan_count
        nan_count += 1
        return float("nan")

    total_burned, _ = evaluate(
        nan_for_all, s, [ignition], treatments_per_step=2, max_steps=3, rng=_rng(), intervention_delay=0
    )
    assert nan_count > 0
    assert total_burned <= 1


def test_apply_treatments_skips_water_and_rock():
    s = create_grid(10, 10, water_fraction=0.1, rock_fraction=0.1, seed=0)
    set_wind(s, speed=10.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.1)
    _apply_treatments(s, lambda state, node: 1.0, budget=s.rows * s.cols, rng=np.random.default_rng(0))
    unburnable_mask = (s.terrain == TerrainType.WATER) | (s.terrain == TerrainType.ROCK)
    assert not (s.state[unburnable_mask] == NodeState.TREATED).any()


def test_apply_treatments_second_pick_is_adjacent_when_adjacency_rewarded():
    s = create_grid(5, 5, seed=0)
    set_wind(s, speed=0.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.1)
    _apply_treatments(
        s, lambda state, node: float(treated_neighbour_count(state, node)), budget=2, rng=np.random.default_rng(0)
    )
    treated = [(int(r), int(c)) for r, c in np.argwhere(s.state == NodeState.TREATED)]
    assert len(treated) == 2
    assert treated[1] in s.neighbours(treated[0]) or treated[0] in s.neighbours(treated[1])


def test_apply_treatments_second_pick_not_adjacent_when_adjacency_penalised():
    s = create_grid(5, 5, seed=0)
    set_wind(s, speed=0.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.1)
    _apply_treatments(
        s, lambda state, node: -float(treated_neighbour_count(state, node)), budget=2, rng=np.random.default_rng(0)
    )
    treated = [(int(r), int(c)) for r, c in np.argwhere(s.state == NodeState.TREATED)]
    assert len(treated) == 2
    assert treated[1] not in s.neighbours(treated[0]) and treated[0] not in s.neighbours(treated[1])


def test_apply_treatments_updates_state_count_cache_during_intra_step_rescoring():
    s = create_grid(5, 5, seed=0)
    set_wind(s, speed=0.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.1)
    precompute_state_counts(s)

    seen_counts: list[tuple[int, int]] = []

    def score_with_counts(state, node):
        seen_counts.append((state.step_treated, state.step_unburned))
        return float(state.step_treated)

    annotate_required_precomputes(score_with_counts, {"total_treated", "total_unburned"})
    _apply_treatments(s, score_with_counts, budget=2, rng=np.random.default_rng(0))

    initial_unburned = s.rows * s.cols
    assert seen_counts[0] == (0, initial_unburned)
    assert (1, initial_unburned - 1) in seen_counts


def test_safe_score_clamps_positive_inf():
    s = _setup()
    assert _safe_score(lambda state, node: float("inf"), s, (0, 0)) == float("-inf")


def test_safe_score_clamps_nan():
    s = _setup()
    assert _safe_score(lambda state, node: float("nan"), s, (0, 0)) == float("-inf")


def test_safe_score_preserves_finite():
    s = _setup()
    assert _safe_score(lambda state, node: 3.14, s, (0, 0)) == 3.14


def test_annotate_required_precomputes_marks_neighbourhood_and_fire_dependencies():
    def func(state, node):
        return 0.0

    annotate_required_precomputes(func, {"burning_neighbour_count", "burning_two_hop_count", "distance_to_fire"})
    assert func._required_precomputes == frozenset({"fire_map", "neighbourhood_maps", "burning_two_hop_map"})
