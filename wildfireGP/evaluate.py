"""
Fitness evaluation for GP individuals and comparison strategies.

evaluate() runs one simulation scenario and returns (total_burned, peak_burning) --- both to be minimised. The graph
passed in is deepcopied internally so the same landscape template can be reused across many evaluations without
mutation.

Treatment selection
-------------------
At each timestep on or after intervention_delay, every UNBURNED node with fuel > 0 (i.e. burnable LAND nodes) is
scored by func(graph, node). The top treatments_per_step nodes by score are treated before spread_step is called.
Non-finite scores (produced by arithmetic like inf - inf in the tree) are clamped to -inf so they sort last and are
never treated preferentially.

Intervention delay
------------------
Real wildfire response involves detection, dispatch, and travel before any resource reaches the fire. intervention_delay
models this as a number of simulation steps during which no treatments are applied. At 100m/cell the default of 3 steps
corresponds to roughly 90 minutes to 2 hours of elapsed real time before initial attack resources arrive --- consistent
with North American initial attack response time targets for fires in remote terrain.

Treatment budget
----------------
treatments_per_step represents the aggregate intervention capacity per timestep across all available resources (aerial
retardant drops, dozer lines, hand crews). At 100m/cell it does not represent individual crew actions but the combined
effect of a coordinated response. The default of 3 reflects a modest initial attack force; larger values represent
heavier resource commitment. Calibration runs on 50x50 grids show that treatments=3 with delay=3 produces a regime
where strategy choice meaningfully changes outcomes --- fire_proximity burns ~10-30% vs ~85% for no_treatment.

Weighted fitness
----------------
total_burned is a count with uniform weight. When a VALUE node attribute is added, total_burned should become
sum(graph.nodes[n][VALUE] for burned nodes) to weight high-value nodes more heavily in the fitness signal.
"""

import copy
import math
from collections.abc import Iterator
from typing import Callable

import networkx as nx
import numpy as np

from wildfireGP.features import (
    precompute_burnable_fire_map,
    precompute_fire_map,
    precompute_reachable_unburned_area,
    precompute_state_counts,
)
from wildfireGP.network import BURN_TIMER, FUEL, STATE, NodeState
from wildfireGP.spread import MAX_BURN_STEPS, spread_step

DEFAULT_TREATMENTS_PER_STEP = 3
DEFAULT_INTERVENTION_DELAY = 3


def init_ignition(graph: nx.Graph, ignition_nodes: list[tuple]) -> None:
    """Set ignition nodes to BURNING state with fuel-proportional burn timers."""
    for node in ignition_nodes:
        graph.nodes[node][STATE] = NodeState.BURNING
        graph.nodes[node][BURN_TIMER] = max(1, math.ceil(graph.nodes[node][FUEL] * MAX_BURN_STEPS))


def simulate(
    graph: nx.Graph,
    func: Callable[[nx.Graph, tuple], float],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> Iterator[tuple[int, nx.Graph]]:
    """
    Run one fire simulation, yielding (step, graph) after each spread step.

    Assumes init_ignition() has already been called. Modifies graph in place. The yielded graph
    is the same object mutated each step — callers that need a snapshot must deepcopy it themselves.
    """
    for step in range(max_steps):
        if not any(graph.nodes[n][STATE] == NodeState.BURNING for n in graph.nodes):
            break
        precompute_fire_map(graph)
        precompute_burnable_fire_map(graph)
        precompute_reachable_unburned_area(graph)
        precompute_state_counts(graph)
        if step >= intervention_delay:
            _apply_treatments(graph, func, treatments_per_step, rng)
        spread_step(graph, rng)
        yield step, graph


def evaluate(
    func: Callable[[nx.Graph, tuple], float],
    graph: nx.Graph,
    ignition_nodes: list[tuple],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> tuple[int, int]:
    """
    Run one simulation scenario and return (total_burned, peak_burning).

    :param func: Compiled GP tree or strategy callable with signature (graph, node) -> float.
        Higher score means higher treatment priority. Must handle any node state gracefully.
    :param graph: Landscape template. Deepcopied internally; the original is never modified.
        Wind and moisture must be set before calling.
    :param ignition_nodes: Nodes to ignite at t=0. Should be UNBURNED LAND nodes.
    :param treatments_per_step: Maximum number of treatments applied per timestep. Represents aggregate
        intervention capacity across all resources, not individual crew actions. Default 3 reflects a
        modest initial attack force at 100m/cell resolution.
    :param max_steps: Maximum simulation steps before terminating regardless of fire state.
    :param rng: NumPy random generator for stochastic spread.
    :param intervention_delay: Number of steps before any treatments are applied. Models detection,
        dispatch, and travel time before resources reach the fire. Default 3 corresponds to roughly
        90 minutes to 2 hours at 100m/cell --- consistent with initial attack response time targets
        for remote terrain in North American fire management.
    :return: (total_burned, peak_burning). Both should be minimised.
    """
    graph = copy.deepcopy(graph)
    init_ignition(graph, ignition_nodes)
    peak_burning = 0
    for _step, _ in simulate(graph, func, treatments_per_step, max_steps, rng, intervention_delay):
        peak_burning = max(peak_burning, sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNING))
    total_burned = sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNED)
    return total_burned, peak_burning


def _apply_treatments(
    graph: nx.Graph, func: Callable[[nx.Graph, tuple], float], budget: int, rng: np.random.Generator
) -> None:
    """
    Apply up to budget treatments to the highest-scoring UNBURNED burnable nodes.

    Candidates are shuffled before sorting so that nodes with equal scores are broken randomly rather than
    by graph insertion order (which would systematically favour low-index nodes — top-left of the grid).
    Python's sort is stable, so the shuffle is the only source of tie-breaking randomness; the score ranking
    is fully preserved above any tie.
    """
    candidates = [n for n in graph.nodes if graph.nodes[n][STATE] == NodeState.UNBURNED and graph.nodes[n][FUEL] > 0.0]
    rng.shuffle(candidates)
    candidates.sort(key=lambda n: _safe_score(func, graph, n), reverse=True)
    for node in candidates[:budget]:
        graph.nodes[node][STATE] = NodeState.TREATED


def _safe_score(func: Callable[[nx.Graph, tuple], float], graph: nx.Graph, node: tuple) -> float:
    score = func(graph, node)
    return score if math.isfinite(score) else float("-inf")
