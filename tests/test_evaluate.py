import numpy as np

from wildfireGP.evaluate import (
    _apply_treatments,
    _safe_score,
    evaluate,
    init_ignition,
    simulate,
)
from wildfireGP.features import (
    distance_to_fire,
    precompute_fire_map,
)
from wildfireGP.network import (
    STATE,
    TERRAIN,
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)


def _setup(rows=7, cols=7, moisture=0.1, wind_speed=10.0, wind_dir=0.0, seed=0):
    g = create_grid(rows, cols, seed=seed)
    set_wind(g, speed=wind_speed, direction=wind_dir)
    set_fuel_moisture(g, moisture=moisture)
    return g


def _no_op(graph, node):
    return 0.0


def _rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Basic return contract
# ---------------------------------------------------------------------------


def test_evaluate_returns_two_ints():
    g = _setup()
    result = evaluate(_no_op, g, [(3, 3)], treatments_per_step=1, max_steps=5, rng=_rng())
    assert len(result) == 2
    assert all(isinstance(v, int) for v in result)


def test_evaluate_no_ignition_returns_zero_burned():
    g = _setup()
    total_burned, peak_burning = evaluate(_no_op, g, [], treatments_per_step=1, max_steps=10, rng=_rng())
    assert total_burned == 0
    assert peak_burning == 0


def test_evaluate_saturated_moisture_fire_does_not_spread():
    g = _setup(moisture=1.0)
    ignition = (3, 3)
    total_burned, _ = evaluate(_no_op, g, [ignition], treatments_per_step=0, max_steps=10, rng=_rng())
    assert total_burned <= 1


def test_evaluate_does_not_modify_original_graph():
    g = _setup()
    states_before = {n: g.nodes[n][STATE] for n in g.nodes}
    evaluate(_no_op, g, [(3, 3)], treatments_per_step=1, max_steps=5, rng=_rng())
    assert {n: g.nodes[n][STATE] for n in g.nodes} == states_before


def test_evaluate_peak_burning_at_least_ignition_count():
    g = _setup()
    ignitions = [(3, 3), (3, 4)]
    _, peak_burning = evaluate(_no_op, g, ignitions, treatments_per_step=0, max_steps=10, rng=_rng())
    assert peak_burning >= len(ignitions)


def test_evaluate_peak_burning_not_greater_than_node_count():
    g = _setup()
    _, peak_burning = evaluate(_no_op, g, [(3, 3)], treatments_per_step=0, max_steps=20, rng=_rng())
    assert peak_burning <= g.number_of_nodes()


# ---------------------------------------------------------------------------
# Treatment effect
# ---------------------------------------------------------------------------


def test_evaluate_zero_treatments_allows_spread():
    g = _setup(moisture=0.05)
    total_burned, _ = evaluate(_no_op, g, [(3, 3)], treatments_per_step=0, max_steps=20, rng=_rng())
    assert total_burned > 1


def test_evaluate_treating_nearest_nodes_reduces_burned_area_smoke():
    g = _setup(moisture=0.05, seed=1)
    ignition = [(3, 3)]

    def no_treatment(graph, node):
        return 0.0

    def nearest_first(graph, node):
        precompute_fire_map(graph)
        d = distance_to_fire(graph, node)
        return -d if d != float("inf") else float("-inf")

    burned_no_treatment, _ = evaluate(no_treatment, g, ignition, treatments_per_step=3, max_steps=15, rng=_rng())
    burned_nearest, _ = evaluate(nearest_first, g, ignition, treatments_per_step=3, max_steps=15, rng=_rng())
    assert burned_nearest <= burned_no_treatment


def test_evaluate_intervention_delay_no_func_calls_before_delay():
    g = _setup(moisture=0.05, seed=0)
    call_count = [0]

    def counting(graph, node):
        call_count[0] += 1
        return 1.0

    evaluate(counting, g, [(3, 3)], treatments_per_step=5, max_steps=5, rng=_rng(), intervention_delay=5)
    assert call_count[0] == 0


def test_evaluate_intervention_delay_func_calls_happen_with_zero_delay():
    g = _setup(moisture=0.05, seed=0)
    call_count = [0]

    def counting(graph, node):
        call_count[0] += 1
        return 1.0

    evaluate(counting, g, [(3, 3)], treatments_per_step=5, max_steps=5, rng=_rng(), intervention_delay=0)
    assert call_count[0] > 0


def test_evaluate_intervention_delay_first_treatment_at_delay_step():
    g = _setup(moisture=0.05, seed=0)
    init_ignition(g, [(3, 3)])
    first_treated_step = None
    for step, sim_graph in simulate(
        g, lambda graph, node: 1.0, treatments_per_step=5, max_steps=10, rng=_rng(), intervention_delay=3
    ):
        if any(sim_graph.nodes[n][STATE] == NodeState.TREATED for n in sim_graph.nodes):
            first_treated_step = step
            break
    assert first_treated_step == 3


def test_evaluate_nan_score_treated_last():
    g = _setup(moisture=1.0)
    ignition = (3, 3)

    nan_count = 0

    def nan_for_all(graph, node):
        nonlocal nan_count
        nan_count += 1
        return float("nan")

    total_burned, _ = evaluate(
        nan_for_all, g, [ignition], treatments_per_step=2, max_steps=3, rng=_rng(), intervention_delay=0
    )
    assert nan_count > 0
    assert total_burned <= 1


def test_apply_treatments_skips_water_and_rock():
    g = create_grid(10, 10, water_fraction=0.1, rock_fraction=0.1, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    _apply_treatments(g, lambda graph, node: 1.0, budget=g.number_of_nodes(), rng=np.random.default_rng(0))
    for n in g.nodes:
        if g.nodes[n][TERRAIN] in (TerrainType.WATER, TerrainType.ROCK):
            from wildfireGP.network import NodeState

            assert g.nodes[n][STATE] != NodeState.TREATED


def test_safe_score_clamps_positive_inf():
    g = _setup()
    assert _safe_score(lambda graph, node: float("inf"), g, (0, 0)) == float("-inf")


def test_safe_score_clamps_nan():
    g = _setup()
    assert _safe_score(lambda graph, node: float("nan"), g, (0, 0)) == float("-inf")


def test_safe_score_preserves_finite():
    g = _setup()
    assert _safe_score(lambda graph, node: 3.14, g, (0, 0)) == 3.14
