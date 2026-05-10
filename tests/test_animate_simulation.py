import math

import dill
import numpy as np
import pytest

from scripts.animate_simulation import _load_strategy, _run_simulation
from wildfireGP.network import (
    BURN_TIMER,
    FUEL,
    STATE,
    NodeState,
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.spread import MAX_BURN_STEPS
from wildfireGP.strategies import no_treatment, score_by_fire_proximity

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph():
    g = create_grid(10, 10, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    ignition = select_ignition_node(g, np.random.default_rng(0))
    g.nodes[ignition][STATE] = NodeState.BURNING
    g.nodes[ignition][BURN_TIMER] = max(1, math.ceil(g.nodes[ignition][FUEL] * MAX_BURN_STEPS))
    return g


@pytest.fixture()
def dill_file(tmp_path):
    path = tmp_path / "hof_0.dill"
    with open(path, "wb") as f:
        dill.dump(no_treatment, f)
    return path


class _FakeArgs:
    def __init__(self, strategy=None, hof=None):
        self.strategy = strategy
        self.hof = hof


# ---------------------------------------------------------------------------
# _load_strategy
# ---------------------------------------------------------------------------


def test_load_strategy_builtin_returns_callable():
    func, _ = _load_strategy(_FakeArgs(strategy="no_treatment"))
    assert callable(func)


def test_load_strategy_builtin_label_matches_name():
    _, label = _load_strategy(_FakeArgs(strategy="score_head_fire"))
    assert label == "score_head_fire"


def test_load_strategy_unknown_name_raises():
    with pytest.raises(SystemExit):
        _load_strategy(_FakeArgs(strategy="does_not_exist"))


def test_load_strategy_dill_returns_callable(dill_file):
    func, _ = _load_strategy(_FakeArgs(hof=dill_file))
    assert callable(func)


def test_load_strategy_dill_label_is_stem(dill_file):
    _, label = _load_strategy(_FakeArgs(hof=dill_file))
    assert label == "hof_0"


# ---------------------------------------------------------------------------
# _run_simulation
# ---------------------------------------------------------------------------


def test_run_simulation_snapshot_count_within_bounds(graph):
    max_steps = 30
    snapshots = _run_simulation(graph, no_treatment, 2, max_steps, np.random.default_rng(0), intervention_delay=0)
    assert 1 <= len(snapshots) <= max_steps + 1


def test_run_simulation_first_snapshot_has_burning_node(graph):
    snapshots = _run_simulation(graph, no_treatment, 2, 30, np.random.default_rng(0), intervention_delay=0)
    assert any(snapshots[0].nodes[n][STATE] == NodeState.BURNING for n in snapshots[0].nodes)


def test_run_simulation_last_snapshot_has_no_burning_nodes_or_hit_max(graph):
    max_steps = 50
    snapshots = _run_simulation(graph, no_treatment, 2, max_steps, np.random.default_rng(0), intervention_delay=0)
    last = snapshots[-1]
    no_fire = not any(last.nodes[n][STATE] == NodeState.BURNING for n in last.nodes)
    assert no_fire or len(snapshots) == max_steps + 1


def test_run_simulation_treatment_reduces_unburned_count(graph):
    snapshots_treated = _run_simulation(
        graph, score_by_fire_proximity, 5, 10, np.random.default_rng(0), intervention_delay=0
    )
    treated_count = sum(
        1 for n in snapshots_treated[-1].nodes if snapshots_treated[-1].nodes[n][STATE] == NodeState.TREATED
    )
    assert treated_count > 0
