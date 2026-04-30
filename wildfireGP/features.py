"""
Per-node and graph-level feature functions for the GP language.

These are the sole building blocks wired into the DEAP primitive set in language.py --- every value the GP language can
read about the world comes through one of these functions, even when the implementation is a direct attribute lookup.
"""

import networkx as nx

from wildfireGP.network import (
    BURN_TIMER,
    ELEVATION,
    FUEL,
    FUEL_MOISTURE,
    SLOPE,
    STATE,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
)


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

def fuel_level(graph: nx.Graph, node: tuple) -> float:
    """Fuel load in [0, 1]. 0 for water and rock."""
    return graph.nodes[node][FUEL]


def elevation(graph: nx.Graph, node: tuple) -> float:
    """Terrain height normalised to [0, 1]."""
    return graph.nodes[node][ELEVATION]


def slope(graph: nx.Graph, node: tuple) -> float:
    """Terrain steepness normalised to [0, 1]."""
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

def burning_neighbor_count(graph: nx.Graph, node: tuple) -> int:
    """Number of directly adjacent nodes that are currently burning."""
    return sum(1 for n in graph.neighbors(node) if graph.nodes[n][STATE] == NodeState.BURNING)


def burned_neighbor_count(graph: nx.Graph, node: tuple) -> int:
    """Number of directly adjacent nodes that have already burned out."""
    return sum(1 for n in graph.neighbors(node) if graph.nodes[n][STATE] == NodeState.BURNED)


def unburned_neighbor_count(graph: nx.Graph, node: tuple) -> int:
    """Number of directly adjacent nodes that are still unburned."""
    return sum(1 for n in graph.neighbors(node) if graph.nodes[n][STATE] == NodeState.UNBURNED)


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

def distance_to_fire(graph: nx.Graph, node: tuple) -> float:
    """Manhattan distance to the nearest burning node. Returns inf if no fire exists."""
    ri, ci = node
    best = float("inf")
    for n in graph.nodes:
        if graph.nodes[n][STATE] == NodeState.BURNING:
            d = abs(n[0] - ri) + abs(n[1] - ci)
            if d < best:
                best = d
    return best


# ---------------------------------------------------------------------------
# Environment (graph-level)
# ---------------------------------------------------------------------------

def wind_speed(graph: nx.Graph) -> float:
    """Wind speed in km/h."""
    return graph.graph[WIND_SPEED]


def wind_direction(graph: nx.Graph) -> float:
    """Wind direction in degrees (0 = north, clockwise)."""
    return graph.graph[WIND_DIRECTION]


def fuel_moisture(graph: nx.Graph) -> float:
    """Relative fuel moisture in [0, 1]. 0 = bone dry, 1 = saturated."""
    return graph.graph[FUEL_MOISTURE]
