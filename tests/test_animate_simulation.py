import math

import dill
import numpy as np
import pytest

from scripts.animate_simulation import _load_strategy, _run_simulation
from wildfireGP.network import (
    NodeState,
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.spread import MAX_BURN_STEPS
from wildfireGP.strategies import random_score, score_by_fire_proximity


@pytest.fixture()
def state():
    s = create_grid(10, 10, seed=0)
    set_wind(s, speed=20.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.2)
    ignition = select_ignition_node(s, np.random.default_rng(0))
    s.state[ignition] = NodeState.BURNING
    s.burn_timer[ignition] = max(1, math.ceil(float(s.fuel[ignition]) * MAX_BURN_STEPS))
    return s


@pytest.fixture()
def dill_file(tmp_path):
    path = tmp_path / "hof_0.dill"
    with open(path, "wb") as f:
        dill.dump(random_score, f)
    return path


class _FakeArgs:
    def __init__(self, strategy=None, hof=None, expr=None, results_dir=None):
        self.strategy = strategy
        self.hof = hof
        self.expr = expr
        self.results_dir = results_dir


def test_load_strategy_builtin_returns_callable():
    func, _ = _load_strategy(_FakeArgs(strategy="random_score"))
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


def test_load_strategy_expr_returns_callable(tmp_path):
    expr = "my_expr"
    pop = [(expr, random_score)]
    with open(tmp_path / "final_population.dill", "wb") as f:
        dill.dump(pop, f)
    func, _ = _load_strategy(_FakeArgs(expr=expr, results_dir=tmp_path))
    assert callable(func)


def test_load_strategy_expr_label_is_truncated_expr(tmp_path):
    expr = "my_expr"
    pop = [(expr, random_score)]
    with open(tmp_path / "final_population.dill", "wb") as f:
        dill.dump(pop, f)
    _, label = _load_strategy(_FakeArgs(expr=expr, results_dir=tmp_path))
    assert expr in label


def test_load_strategy_expr_without_results_dir_raises():
    with pytest.raises(SystemExit):
        _load_strategy(_FakeArgs(expr="my_expr"))


def test_run_simulation_snapshot_count_within_bounds(state):
    max_steps = 30
    snapshots = _run_simulation(state, random_score, 2, max_steps, np.random.default_rng(0), intervention_delay=0)
    assert 1 <= len(snapshots) <= max_steps + 1


def test_run_simulation_first_snapshot_has_burning_cell(state):
    snapshots = _run_simulation(state, random_score, 2, 30, np.random.default_rng(0), intervention_delay=0)
    assert (snapshots[0].state == NodeState.BURNING).any()


def test_run_simulation_last_snapshot_has_no_burning_cells_or_hit_max(state):
    max_steps = 50
    snapshots = _run_simulation(state, random_score, 2, max_steps, np.random.default_rng(0), intervention_delay=0)
    last = snapshots[-1]
    no_fire = not (last.state == NodeState.BURNING).any()
    assert no_fire or len(snapshots) == max_steps + 1


def test_run_simulation_treatments_are_applied(state):
    snapshots = _run_simulation(state, score_by_fire_proximity, 5, 10, np.random.default_rng(0), intervention_delay=0)
    treated_count = int((snapshots[-1].state == NodeState.TREATED).sum())
    assert treated_count > 0
