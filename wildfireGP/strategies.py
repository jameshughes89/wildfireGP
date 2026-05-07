"""
Baseline heuristic strategies for wildfire suppression resource allocation.

Each strategy is a callable (graph, node) -> float compatible with evaluate(). Higher score means higher treatment
priority. These serve as comparison baselines for GP-evolved strategies.

Strategy ladder (weakest to strongest expected performance):
    no_treatment < random_score < score_by_fuel / score_by_burning_neighbors
        < score_by_fire_proximity < score_indirect_attack / score_ridgeline / score_head_fire

If GP cannot outperform score_by_fire_proximity it is not producing useful strategies.

The domain-informed strategies (score_indirect_attack, score_ridgeline, score_head_fire) are derived from standard
operational doctrine for wildland fire suppression. The canonical reference is the NWCG Fireline Handbook, which
codifies direct and indirect attack tactics, anchor point selection, and head-fire defence used by suppression crews.
Topographic effects (slope acceleration, ridgeline anchor points) follow from Rothermel (1972), which is cited in
network.py.

References
----------
National Wildfire Coordinating Group. (2004). Fireline Handbook. NWCG Handbook 3, PMS 410-1.
Rothermel, R.C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service
    Research Paper INT-115.
"""

import random

import networkx as nx

from wildfireGP.features import (
    burning_neighbour_count,
    distance_to_fire,
    elevation,
    fuel_level,
    mean_neighbour_fuel,
    slope,
    wind_fire_alignment,
)


def no_treatment(graph: nx.Graph, node: tuple) -> float:
    """
    Score every node equally --- nothing is ever preferentially treated. Lower bound baseline.
    """
    return 0.0


def random_score(graph: nx.Graph, node: tuple) -> float:
    """
    Assign a random priority score --- treatment order is arbitrary. Beats no_treatment by luck.
    """
    return random.random()


def score_by_fuel(graph: nx.Graph, node: tuple) -> float:
    """
    Prioritise nodes with the highest fuel load --- remove the most burnable material first.
    """
    return fuel_level(graph, node)


def score_by_fire_proximity(graph: nx.Graph, node: tuple) -> float:
    """
    Prioritise nodes closest to the active fire front (direct attack).
    """
    return -distance_to_fire(graph, node)


def score_by_burning_neighbors(graph: nx.Graph, node: tuple) -> float:
    """
    Prioritise nodes already surrounded by burning neighbours --- ring-buffer / direct defence.
    """
    return float(burning_neighbour_count(graph, node))


def score_indirect_attack(graph: nx.Graph, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """
    Prioritise fuel-rich ground just ahead of the fire front (indirect attack).

    Firefighters build control lines ahead of the active edge rather than fighting flame directly. Scores nodes by the
    fuel load of their neighbourhood weighted by proximity --- clearing a high-fuel zone before the fire arrives is more
    effective than reacting at the burning edge.

    Returns 0.0 outside [min_distance, max_distance]. If the fire is already adjacent (< min_distance) it is too late
    for indirect prep; if it is beyond max_distance the node is unlikely to be threatened imminently.

    :param min_distance: Minimum hop distance to fire to engage (default 2 = 200m at 100m/cell).
    :param max_distance: Maximum hop distance to fire to engage (default 10 = 1km at 100m/cell).

    Note: defaults are tuneable heuristics, not values derived from a specific cited source.
    """
    d = distance_to_fire(graph, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return mean_neighbour_fuel(graph, node) / (1.0 + d)


def score_ridgeline(graph: nx.Graph, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """
    Prioritise topographic high points within engagement range of the fire (ridgeline defence).

    Fire spreads fastest uphill. Holding a ridgeline or steep slope denies the fire an acceleration path and provides a
    natural anchor for control lines. Only nodes within [min_distance, max_distance] of the fire are considered ---
    committing crews to a distant ridge is premature, and establishing a ridgeline anchor when the fire is already
    adjacent is too late.

    :param min_distance: Minimum hop distance to fire to engage (default 2 = 200m at 100m/cell).
    :param max_distance: Maximum hop distance to fire to engage (default 10 = 1km at 100m/cell).

    Note: defaults are tuneable heuristics, not values derived from a specific cited source.
    """
    d = distance_to_fire(graph, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return elevation(graph, node) + slope(graph, node)


def score_head_fire(graph: nx.Graph, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """
    Prioritise nodes downwind and within engagement range of the fire (head fire defence).

    The head of a fire --- its downwind front --- advances fastest and is hardest to stop once established. Scores nodes
    by wind alignment with the fire direction weighted by proximity, defending the path the fire is most aggressively
    pursuing. Only nodes within [min_distance, max_distance] are considered --- head fire positioning is proactive, not
    reactive; if the fire is already adjacent the window for effective positioning has passed.

    :param min_distance: Minimum hop distance to fire to engage (default 2 = 200m at 100m/cell).
    :param max_distance: Maximum hop distance to fire to engage (default 10 = 1km at 100m/cell).

    Note: defaults are tuneable heuristics, not values derived from a specific cited source.
    """
    d = distance_to_fire(graph, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return wind_fire_alignment(graph, node) / (1.0 + d)


ALL_STRATEGIES = [
    no_treatment,
    random_score,
    score_by_fuel,
    score_by_fire_proximity,
    score_by_burning_neighbors,
    score_indirect_attack,
    score_ridgeline,
    score_head_fire,
]
