"""
Baseline heuristic strategies for wildfire suppression resource allocation.

Each strategy is a callable (graph, node) -> float compatible with evaluate(). Higher score means higher
treatment priority. These serve as comparison baselines for GP-evolved strategies.

Strategy ladder (weakest to strongest expected performance):
    no_treatment < random_score < score_by_fuel / score_by_burning_neighbors
        < score_by_fire_proximity < score_indirect_attack / score_ridgeline / score_head_fire

If GP cannot outperform score_by_fire_proximity it is not producing useful strategies.
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
    """Score every node equally — nothing is ever preferentially treated. Lower bound baseline."""
    return 0.0


def random_score(graph: nx.Graph, node: tuple) -> float:
    """Assign a random priority score — treatment order is arbitrary. Beats no_treatment by luck."""
    return random.random()


def score_by_fuel(graph: nx.Graph, node: tuple) -> float:
    """Prioritise nodes with the highest fuel load — remove the most burnable material first."""
    return fuel_level(graph, node)


def score_by_fire_proximity(graph: nx.Graph, node: tuple) -> float:
    """Prioritise nodes closest to the active fire front (direct attack)."""
    return -distance_to_fire(graph, node)


def score_by_burning_neighbors(graph: nx.Graph, node: tuple) -> float:
    """Prioritise nodes already surrounded by burning neighbours — ring-buffer / direct defence."""
    return float(burning_neighbour_count(graph, node))


def score_indirect_attack(graph: nx.Graph, node: tuple) -> float:
    """
    Prioritise fuel-rich ground just ahead of the fire front (indirect attack).

    Firefighters build control lines ahead of the active edge rather than fighting flame directly.
    Scores nodes by the fuel load of their neighbourhood weighted by proximity — clearing a
    high-fuel zone before the fire arrives is more effective than reacting at the burning edge.
    """
    return mean_neighbour_fuel(graph, node) / (1.0 + distance_to_fire(graph, node))


def score_ridgeline(graph: nx.Graph, node: tuple) -> float:
    """
    Prioritise topographic high points (ridgeline defence).

    Fire spreads fastest uphill. Holding a ridgeline or steep slope denies the fire an
    acceleration path and provides a natural anchor for control lines.
    """
    return elevation(graph, node) + slope(graph, node)


def score_head_fire(graph: nx.Graph, node: tuple) -> float:
    """
    Prioritise nodes downwind and close to the fire (head fire defence).

    The head of a fire — its downwind front — advances fastest and is hardest to stop once
    established. Scores nodes by wind alignment with the fire direction weighted by proximity,
    defending the path the fire is most aggressively pursuing.
    """
    return wind_fire_alignment(graph, node) / (1.0 + distance_to_fire(graph, node))


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
