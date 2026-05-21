"""
Fitness evaluation for GP individuals and comparison strategies.

:func:`evaluate` runs one simulation scenario and returns ``(total_burned, peak_burning)`` --- both to be minimised. The
landscape state passed in is copied internally so the same template can be reused across many evaluations without
mutation.

Treatment selection
-------------------
At each timestep on or after ``intervention_delay``, every UNBURNED node with fuel > 0 (i.e. burnable LAND nodes) is
scored by ``func(state, node)``. The top ``treatments_per_step`` nodes by score are treated before :func:`spread_step`
is called. Non-finite scores (produced by arithmetic like inf - inf in the tree) are clamped to -inf so they sort last
and are never treated preferentially.

Intervention delay
------------------
``intervention_delay`` models detection, dispatch, and travel before initial attack resources arrive. Default 3 steps
correspond to roughly 90 minutes to 2 hours at 100m/cell --- consistent with North American initial attack response
time targets for fires in remote terrain.

Treatment budget
----------------
``treatments_per_step`` is the aggregate intervention capacity per timestep across all available resources (aerial
retardant drops, dozer lines, hand crews). Default 3 reflects a modest initial attack force.

Weighted fitness
----------------
``total_burned`` is a count with uniform weight. When a VALUE node attribute is added, ``total_burned`` should become
``sum(state.value[n] for burned nodes)`` to weight high-value nodes more heavily in the fitness signal.
"""

import math
from collections.abc import Iterator
from typing import Callable

import numpy as np

from wildfireGP.features import (
    precompute_burnable_fire_map,
    precompute_fire_map,
    precompute_reachable_unburned_area,
    precompute_state_counts,
)
from wildfireGP.network import NodeState, SimState
from wildfireGP.spread import MAX_BURN_STEPS, spread_step

DEFAULT_TREATMENTS_PER_STEP = 3
DEFAULT_INTERVENTION_DELAY = 3


def init_ignition(state: SimState, ignition_nodes: list[tuple]) -> None:
    """Set ignition nodes to BURNING state with fuel-proportional burn timers."""
    for node in ignition_nodes:
        state.state[node] = NodeState.BURNING
        state.burn_timer[node] = max(1, math.ceil(float(state.fuel[node]) * MAX_BURN_STEPS))


def simulate(
    state: SimState,
    func: Callable[[SimState, tuple], float],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> Iterator[tuple[int, SimState]]:
    """
    Run one fire simulation, yielding ``(step, state)`` after each spread step.

    Assumes :func:`init_ignition` has already been called. Modifies state in place. The yielded state is the same
    object mutated each step --- callers that need a snapshot must copy it themselves.
    """
    for step in range(max_steps):
        if not (state.state == NodeState.BURNING).any():
            break
        if step >= intervention_delay:
            precompute_fire_map(state)
            precompute_burnable_fire_map(state)
            precompute_reachable_unburned_area(state)
            precompute_state_counts(state)
            _apply_treatments(state, func, treatments_per_step, rng)
        spread_step(state, rng)
        yield step, state


def evaluate(
    func: Callable[[SimState, tuple], float],
    state: SimState,
    ignition_nodes: list[tuple],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> tuple[int, int]:
    """
    Run one simulation scenario and return ``(total_burned, peak_burning)``.

    :param func: Compiled GP tree or strategy callable ``(state, node) -> float``. Higher score = higher priority.
    :param state: Landscape template. Copied internally; the original is never modified. Wind and moisture must be set
        before calling.
    :param ignition_nodes: Nodes to ignite at t=0. Should be UNBURNED LAND nodes.
    :param treatments_per_step: Maximum number of treatments applied per timestep.
    :param max_steps: Maximum simulation steps before terminating regardless of fire state.
    :param rng: NumPy random generator for stochastic spread.
    :param intervention_delay: Number of steps before any treatments are applied.
    :return: ``(total_burned, peak_burning)``. Both should be minimised.
    """
    state = state.copy()
    init_ignition(state, ignition_nodes)
    peak_burning = int((state.state == NodeState.BURNING).sum())
    for _step, _ in simulate(state, func, treatments_per_step, max_steps, rng, intervention_delay):
        peak_burning = max(peak_burning, int((state.state == NodeState.BURNING).sum()))
    total_burned = int((state.state == NodeState.BURNED).sum())
    return total_burned, peak_burning


def _apply_treatments(
    state: SimState, func: Callable[[SimState, tuple], float], budget: int, rng: np.random.Generator
) -> None:
    """
    Apply up to ``budget`` treatments to the highest-scoring UNBURNED burnable nodes.

    Candidates are scored once up front, then selected one at a time. After each placement only the Moore-neighbours of
    the just-treated node are rescored --- they are the only nodes whose treated-neighbour features can have changed.
    Candidates are shuffled before the initial sort so equal-scoring nodes are broken randomly; timsort's stability
    preserves that order on re-sorts.
    """
    candidate_arr = np.argwhere((state.state == NodeState.UNBURNED) & (state.fuel > 0.0))
    candidates = [(int(r), int(c)) for r, c in candidate_arr]
    rng.shuffle(candidates)
    scores = {n: _safe_score(func, state, n) for n in candidates}
    candidate_set = set(candidates)
    candidates.sort(key=lambda n: scores[n], reverse=True)
    for _ in range(min(budget, len(candidates))):
        node = candidates.pop(0)
        candidate_set.discard(node)
        state.state[node] = NodeState.TREATED
        to_rescore = [n for n in state.neighbours(node) if n in candidate_set]
        if to_rescore:
            for n in to_rescore:
                scores[n] = _safe_score(func, state, n)
            candidates.sort(key=lambda n: scores[n], reverse=True)


def _safe_score(func: Callable[[SimState, tuple], float], state: SimState, node: tuple) -> float:
    score = func(state, node)
    return score if math.isfinite(score) else float("-inf")
