"""Fitness evaluation for GP individuals and comparison strategies.

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
    precompute_betweenness_dynamic,
    precompute_betweenness_static,
    precompute_burnable_fire_map,
    precompute_burning_two_hop_map,
    precompute_fire_distance_map,
    precompute_fire_map,
    precompute_mtt_pathway_count,
    precompute_neighbourhood_maps,
    precompute_reachable_unburned_area,
    precompute_state_counts,
    update_neighbourhood_maps_after_treatment,
)
from wildfireGP.network import GraphState, NodeState
from wildfireGP.spread import MAX_BURN_STEPS, spread_step

DEFAULT_TREATMENTS_PER_STEP = 3
DEFAULT_INTERVENTION_DELAY = 3
DEFAULT_MIN_TREATMENT_DISTANCE = 0
ALL_PRECOMPUTES = frozenset(
    {
        "fire_map",
        "fire_distance_map",
        "burnable_fire_map",
        "reachable_unburned_area",
        "state_counts",
        "neighbourhood_maps",
        "burning_two_hop_map",
        "betweenness_static",
        "betweenness_dynamic",
        "mtt_pathway_count",
    }
)

_FEATURE_PRECOMPUTE_MAP = {
    "mean_neighbour_elevation": {"neighbourhood_maps"},
    "mean_neighbour_fuel": {"neighbourhood_maps"},
    "burning_neighbour_count": {"neighbourhood_maps"},
    "burning_two_hop_count": {"burning_two_hop_map"},
    "unburned_neighbour_count": {"neighbourhood_maps"},
    "unburnable_neighbour_count": {"neighbourhood_maps"},
    "has_treated_neighbour": {"neighbourhood_maps"},
    "treated_neighbour_count": {"neighbourhood_maps"},
    "distance_to_fire": {"fire_distance_map"},
    "elevation_delta_to_fire": {"fire_map"},
    "wind_fire_alignment": {"fire_map"},
    "burnable_distance_to_fire": {"burnable_fire_map"},
    "reachable_unburned_area": {"reachable_unburned_area"},
    "total_burning": {"state_counts"},
    "total_burned": {"state_counts"},
    "total_unburned": {"state_counts"},
    "total_treated": {"state_counts"},
    "betweenness_static": {"betweenness_static"},
    "betweenness_dynamic": {"betweenness_dynamic"},
    "mtt_pathway_count": {"mtt_pathway_count"},
}


def init_ignition(state: GraphState, ignition_nodes: list[tuple]) -> None:
    """Set ignition nodes to BURNING state with fuel-proportional burn timers."""
    for node in ignition_nodes:
        state.state[node] = NodeState.BURNING
        state.burn_timer[node] = max(1, math.ceil(float(state.fuel[node]) * MAX_BURN_STEPS))


def simulate(
    state: GraphState,
    func: Callable[[GraphState, tuple], float],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
    min_treatment_distance: int = DEFAULT_MIN_TREATMENT_DISTANCE,
) -> Iterator[tuple[int, GraphState]]:
    """Run one fire simulation, yielding ``(step, state)`` after each spread step.

    Assumes :func:`init_ignition` has already been called. Modifies state in place. The yielded state is the same
    object mutated each step --- callers that need a snapshot must copy it themselves.

    ``min_treatment_distance`` enforces a hard exclusion zone around active fire: candidate cells whose chessboard
    distance to the nearest burning cell is less than this value are removed from the treatment pool. Default 0 models
    aerial retardant (adjacent treatment allowed); positive values model ground-crew safety zones.
    """
    required_precomputes = getattr(func, "_required_precomputes", ALL_PRECOMPUTES)
    if min_treatment_distance > 0:
        required_precomputes = required_precomputes | {"fire_distance_map"}
    for step in range(max_steps):
        if not (state.state == NodeState.BURNING).any():
            break
        if step >= intervention_delay:
            _run_required_precomputes(state, required_precomputes)
            _apply_treatments(state, func, treatments_per_step, rng, min_treatment_distance)
        spread_step(state, rng)
        yield step, state


def evaluate(
    func: Callable[[GraphState, tuple], float],
    state: GraphState,
    ignition_nodes: list[tuple],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
    min_treatment_distance: int = DEFAULT_MIN_TREATMENT_DISTANCE,
) -> tuple[int, int]:
    """Run one simulation scenario and return ``(total_burned, peak_burning)``.

    :param func: Compiled GP tree or strategy callable ``(state, node) -> float``. Higher score = higher priority.
    :param state: Landscape template. Copied internally; the original is never modified. Wind and moisture must be set
        before calling.
    :param ignition_nodes: Nodes to ignite at t=0. Should be UNBURNED LAND nodes.
    :param treatments_per_step: Maximum number of treatments applied per timestep.
    :param max_steps: Maximum simulation steps before terminating regardless of fire state.
    :param rng: NumPy random generator for stochastic spread.
    :param intervention_delay: Number of steps before any treatments are applied.
    :param min_treatment_distance: Hard exclusion zone around active fire. Default 0 = adjacent treatment allowed
        (aerial retardant). Positive values exclude cells within that chessboard distance of any burning cell from
        the treatment candidate pool (ground-crew safety zone).
    :return: ``(total_burned, peak_burning)``. Both should be minimised.
    """
    state = state.copy()
    init_ignition(state, ignition_nodes)
    peak_burning = int((state.state == NodeState.BURNING).sum())
    for _step, _ in simulate(
        state, func, treatments_per_step, max_steps, rng, intervention_delay, min_treatment_distance
    ):
        peak_burning = max(peak_burning, int((state.state == NodeState.BURNING).sum()))
    total_burned = int((state.state == NodeState.BURNED).sum())
    return total_burned, peak_burning


def _apply_treatments(
    state: GraphState,
    func: Callable[[GraphState, tuple], float],
    budget: int,
    rng: np.random.Generator,
    min_treatment_distance: int = DEFAULT_MIN_TREATMENT_DISTANCE,
) -> None:
    """Apply up to ``budget`` treatments to the highest-scoring UNBURNED burnable nodes.

    Candidates are scored once up front, then selected one at a time. After each placement only the Moore-neighbours of
    the just-treated node are rescored --- they are the only nodes whose treated-neighbour features can have changed.
    Candidates are shuffled before the initial sort so equal-scoring nodes are broken randomly; timsort's stability
    preserves that order on re-sorts.

    When ``min_treatment_distance > 0``, cells within that chessboard distance of any active fire cell are excluded
    from the candidate pool. The GP/strategy score for those cells is computed normally but never translates into a
    placement: the filter is a hard constraint enforced at the placement step, not a penalty on the score.
    """
    mask = (state.state == NodeState.UNBURNED) & (state.fuel > 0.0)
    if min_treatment_distance > 0 and state.fire_distance_map is not None:
        mask = mask & (state.fire_distance_map >= min_treatment_distance)
    candidate_arr = np.argwhere(mask)
    candidates = [(int(r), int(c)) for r, c in candidate_arr]
    rng.shuffle(candidates)
    scores = {n: _safe_score(func, state, n) for n in candidates}
    required_precomputes = getattr(func, "_required_precomputes", ALL_PRECOMPUTES)
    candidate_set = set(candidates)
    candidates.sort(key=lambda n: scores[n], reverse=True)
    for _ in range(min(budget, len(candidates))):
        node = candidates.pop(0)
        candidate_set.discard(node)
        state.state[node] = NodeState.TREATED
        if "state_counts" in required_precomputes:
            state.step_treated += 1
            state.step_unburned -= 1
        update_neighbourhood_maps_after_treatment(state, node)
        to_rescore = [n for n in state.neighbours(node) if n in candidate_set]
        if to_rescore:
            for n in to_rescore:
                scores[n] = _safe_score(func, state, n)
            candidates.sort(key=lambda n: scores[n], reverse=True)


def _safe_score(func: Callable[[GraphState, tuple], float], state: GraphState, node: tuple) -> float:
    score = func(state, node)
    return score if math.isfinite(score) else float("-inf")


_PRECOMPUTE_DISPATCH = {
    "burnable_fire_map": precompute_burnable_fire_map,
    "reachable_unburned_area": precompute_reachable_unburned_area,
    "state_counts": precompute_state_counts,
    "neighbourhood_maps": precompute_neighbourhood_maps,
    "burning_two_hop_map": precompute_burning_two_hop_map,
    "betweenness_static": precompute_betweenness_static,
    "betweenness_dynamic": precompute_betweenness_dynamic,
    "mtt_pathway_count": precompute_mtt_pathway_count,
}


def _run_required_precomputes(state: GraphState, required_precomputes: frozenset[str]) -> None:
    if "fire_map" in required_precomputes:
        precompute_fire_map(state)
    elif "fire_distance_map" in required_precomputes:
        precompute_fire_distance_map(state)
    for name, fn in _PRECOMPUTE_DISPATCH.items():
        if name in required_precomputes:
            fn(state)


def annotate_required_precomputes(func, feature_names: set[str] | list[str] | tuple[str, ...]):
    required = set()
    for feature_name in feature_names:
        required.update(_FEATURE_PRECOMPUTE_MAP.get(feature_name, ()))
    func._required_precomputes = frozenset(required)
    return func
