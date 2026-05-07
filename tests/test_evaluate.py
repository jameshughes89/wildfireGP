import numpy as np

from wildfireGP.evaluate import _safe_score, evaluate
from wildfireGP.features import distance_to_fire, precompute_fire_map
from wildfireGP.network import STATE, create_grid, set_fuel_moisture, set_wind

_RNG = np.random.default_rng(0)


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


def test_evaluate_treating_nearest_nodes_reduces_burned_area():
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


def test_evaluate_intervention_delay_no_treatments_applied_before_delay():
    g = _setup(moisture=0.05, seed=0)
    treated_steps = []

    def record_treated(graph, node):
        from wildfireGP.network import NodeState

        treated_steps.append(sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.TREATED))
        return 1.0

    evaluate(record_treated, g, [(3, 3)], treatments_per_step=5, max_steps=10, rng=_rng(), intervention_delay=3)
    assert treated_steps[0] == 0
    assert treated_steps[1] == 0
    assert treated_steps[2] == 0


def test_evaluate_intervention_delay_treatments_applied_after_delay():
    g = _setup(moisture=0.05, seed=0)
    evaluate(_no_op, g, [(3, 3)], treatments_per_step=5, max_steps=10, rng=_rng(), intervention_delay=3)
    from wildfireGP.network import NodeState

    treated = sum(1 for n in g.nodes if g.nodes[n][STATE] == NodeState.TREATED)
    assert treated == 0


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


def test_safe_score_clamps_positive_inf():
    g = _setup()
    assert _safe_score(lambda graph, node: float("inf"), g, (0, 0)) == float("-inf")


def test_safe_score_clamps_nan():
    g = _setup()
    assert _safe_score(lambda graph, node: float("nan"), g, (0, 0)) == float("-inf")


def test_safe_score_preserves_finite():
    g = _setup()
    assert _safe_score(lambda graph, node: 3.14, g, (0, 0)) == 3.14
