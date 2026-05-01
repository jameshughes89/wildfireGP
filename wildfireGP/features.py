"""
Per-node and graph-level feature functions for the GP language.

These are the sole building blocks wired into the DEAP primitive set in language.py --- every value the GP language can
read about the world comes through one of these functions, even when the implementation is a direct attribute lookup.
"""

import math

import networkx as nx

from wildfireGP.network import (
    BURN_TIMER,
    ELEVATION,
    FUEL,
    FUEL_MOISTURE,
    SLOPE,
    STATE,
    TERRAIN,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
    TerrainType,
)


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

def fuel_level(graph: nx.Graph, node: tuple) -> float:
    return graph.nodes[node][FUEL]


def elevation(graph: nx.Graph, node: tuple) -> float:
    return graph.nodes[node][ELEVATION]


def slope(graph: nx.Graph, node: tuple) -> float:
    return graph.nodes[node][SLOPE]


# ---------------------------------------------------------------------------
# Fire state
# ---------------------------------------------------------------------------

def is_unburned(graph: nx.Graph, node: tuple) -> bool:
    return graph.nodes[node][STATE] == NodeState.UNBURNED


def is_burning(graph: nx.Graph, node: tuple) -> bool:
    return graph.nodes[node][STATE] == NodeState.BURNING


def is_burned(graph: nx.Graph, node: tuple) -> bool:
    return graph.nodes[node][STATE] == NodeState.BURNED


def is_treated(graph: nx.Graph, node: tuple) -> bool:
    return graph.nodes[node][STATE] == NodeState.TREATED


def burn_steps_remaining(graph: nx.Graph, node: tuple) -> int:
    return graph.nodes[node][BURN_TIMER]


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------

def burning_neighbour_count(graph: nx.Graph, node: tuple) -> int:
    return sum(1 for n in graph.neighbors(node) if graph.nodes[n][STATE] == NodeState.BURNING)


def unburned_neighbour_count(graph: nx.Graph, node: tuple) -> int:
    return sum(1 for n in graph.neighbors(node) if graph.nodes[n][STATE] == NodeState.UNBURNED)


def unburnable_neighbour_count(graph: nx.Graph, node: tuple) -> int:
    count = 0
    for n in graph.neighbors(node):
        state = graph.nodes[n][STATE]
        terrain = graph.nodes[n][TERRAIN]
        if state in (NodeState.BURNED, NodeState.TREATED) or terrain in (TerrainType.WATER, TerrainType.ROCK):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

def distance_to_fire(graph: nx.Graph, node: tuple) -> float:
    ri, ci = node
    best = float("inf")
    for n in graph.nodes:
        if graph.nodes[n][STATE] == NodeState.BURNING:
            d = abs(n[0] - ri) + abs(n[1] - ci)
            if d < best:
                best = d
    return best


def wind_fire_alignment(graph: nx.Graph, node: tuple) -> float:
    """
    Cosine similarity between the wind direction and the vector from the nearest burning node to this node.
    +1: directly downwind of nearest fire. -1: directly upwind. 0: crosswind or no fire.
    """
    ri, ci = node
    best_dist = float("inf")
    best_fire = None
    for n in graph.nodes:
        if graph.nodes[n][STATE] == NodeState.BURNING:
            d = abs(n[0] - ri) + abs(n[1] - ci)
            if d < best_dist:
                best_dist = d
                best_fire = n

    if best_fire is None or best_dist == 0:
        return 0.0

    fi, fj = best_fire
    north = fi - ri   # positive = fire is south, vector points north (row decreases going north)
    east = ci - fj    # positive = fire is west, vector points east
    mag = math.sqrt(north ** 2 + east ** 2)

    wind_toward_rad = math.radians(graph.graph[WIND_DIRECTION]) + math.pi
    return (north * math.cos(wind_toward_rad) + east * math.sin(wind_toward_rad)) / mag


# ---------------------------------------------------------------------------
# Environment (graph-level)
# ---------------------------------------------------------------------------

def wind_speed(graph: nx.Graph) -> float:
    return graph.graph[WIND_SPEED]


def wind_direction(graph: nx.Graph) -> float:
    return graph.graph[WIND_DIRECTION]


def fuel_moisture(graph: nx.Graph) -> float:
    return graph.graph[FUEL_MOISTURE]


# ---------------------------------------------------------------------------
# Whole-graph state
# ---------------------------------------------------------------------------

def total_burning(graph: nx.Graph) -> int:
    return sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNING)


def total_burned(graph: nx.Graph) -> int:
    return sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNED)


def total_unburned(graph: nx.Graph) -> int:
    return sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.UNBURNED)


def total_treated(graph: nx.Graph) -> int:
    return sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.TREATED)
