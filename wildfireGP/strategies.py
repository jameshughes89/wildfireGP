"""
Baseline heuristic strategies for wildfire suppression resource allocation.

Each strategy is a callable ``(state, node) -> float`` compatible with :func:`wildfireGP.evaluate.evaluate`. Higher
score means higher treatment priority.

Strategy ladder (weakest to strongest expected performance):
    random_score < score_by_fuel / score_by_burning_neighbors
        < score_by_fire_proximity < score_indirect_attack / score_ridgeline / score_head_fire
        < score_fire_run

If GP cannot outperform :func:`score_by_fire_proximity` it is not producing useful strategies.

True no-treatment baseline
--------------------------
There is no ``no_treatment`` strategy function. A strategy that scores all nodes equally is indistinguishable from
``random_score`` after the shuffle in ``_apply_treatments`` --- treatments are still placed, just randomly. The true
lower bound is a run with ``treatments_per_step=0``, which ``compare_strategies.py`` includes automatically as the
``no_treatment`` row.

References
----------
National Wildfire Coordinating Group. (2004). Fireline Handbook. NWCG Handbook 3, PMS 410-1.
Rothermel, R.C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service
    Research Paper INT-115.
"""

import random

from wildfireGP.evaluate import annotate_required_precomputes
from wildfireGP.features import (
    burning_neighbour_count,
    distance_to_fire,
    elevation,
    fuel_level,
    has_treated_neighbour,
    mean_neighbour_fuel,
    slope,
    wind_fire_alignment,
)
from wildfireGP.network import GraphState

ANCHOR_WEIGHT = 0.1


def random_score(state: GraphState, node: tuple) -> float:
    """Assign a random priority score --- treatment order is arbitrary."""
    return random.random()


def score_by_fuel(state: GraphState, node: tuple) -> float:
    """Prioritise nodes with the highest fuel load."""
    return fuel_level(state, node) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_by_fire_proximity(state: GraphState, node: tuple) -> float:
    """Prioritise nodes closest to the active fire front (direct attack)."""
    return -distance_to_fire(state, node) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_by_burning_neighbors(state: GraphState, node: tuple) -> float:
    """Prioritise nodes already surrounded by burning neighbours (direct defence)."""
    return float(burning_neighbour_count(state, node)) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_indirect_attack(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise fuel-rich ground just ahead of the fire front (indirect attack)."""
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return mean_neighbour_fuel(state, node) / (1.0 + d) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_ridgeline(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise topographic high points within engagement range of the fire (ridgeline defence)."""
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return elevation(state, node) + slope(state, node) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_fire_run(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise fuel-rich nodes downwind of the fire within engagement range."""
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return fuel_level(state, node) * wind_fire_alignment(state, node) / (
        1.0 + d
    ) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_head_fire(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise nodes downwind and within engagement range of the fire (head fire defence)."""
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return wind_fire_alignment(state, node) / (1.0 + d) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


ALL_STRATEGIES = [
    random_score,
    score_by_fuel,
    score_by_fire_proximity,
    score_by_burning_neighbors,
    score_indirect_attack,
    score_ridgeline,
    score_head_fire,
    score_fire_run,
]

annotate_required_precomputes(random_score, [])
annotate_required_precomputes(score_by_fuel, ["has_treated_neighbour"])
annotate_required_precomputes(score_by_fire_proximity, ["distance_to_fire", "has_treated_neighbour"])
annotate_required_precomputes(score_by_burning_neighbors, ["burning_neighbour_count", "has_treated_neighbour"])
annotate_required_precomputes(
    score_indirect_attack, ["distance_to_fire", "mean_neighbour_fuel", "has_treated_neighbour"]
)
annotate_required_precomputes(score_ridgeline, ["distance_to_fire", "has_treated_neighbour"])
annotate_required_precomputes(score_fire_run, ["distance_to_fire", "wind_fire_alignment", "has_treated_neighbour"])
annotate_required_precomputes(score_head_fire, ["distance_to_fire", "wind_fire_alignment", "has_treated_neighbour"])
