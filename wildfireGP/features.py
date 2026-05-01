"""
Per-node and graph-level feature functions for the GP language.

These are the sole building blocks wired into the DEAP primitive set in language.py --- every value the GP language can
read about the world comes through one of these functions, even when the implementation is a direct attribute lookup.
"""

import math
from collections import deque

import networkx as nx

from wildfireGP.network import (
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


def mean_neighbour_elevation(graph: nx.Graph, node: tuple) -> float:
    neighbours = list(graph.neighbors(node))
    return sum(graph.nodes[n][ELEVATION] for n in neighbours) / len(neighbours)


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------


def mean_neighbour_fuel(graph: nx.Graph, node: tuple) -> float:
    neighbours = list(graph.neighbors(node))
    return sum(graph.nodes[n][FUEL] for n in neighbours) / len(neighbours)


def burning_neighbour_count(graph: nx.Graph, node: tuple) -> int:
    return sum(1 for n in graph.neighbors(node) if graph.nodes[n][STATE] == NodeState.BURNING)


def burning_two_hop_count(graph: nx.Graph, node: tuple) -> int:
    one_hop = set(graph.neighbors(node))
    two_hop: set[tuple] = set()
    for neighbour in one_hop:
        two_hop.update(graph.neighbors(neighbour))
    two_hop.discard(node)
    two_hop.difference_update(one_hop)
    return sum(1 for n in two_hop if graph.nodes[n][STATE] == NodeState.BURNING)


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
# Spatial — requires precompute_fire_map to be called first each simulation step
# ---------------------------------------------------------------------------

_NEAREST_FIRE = "nearest_fire"
_FIRE_DISTANCE = "fire_distance"


def precompute_fire_map(graph: nx.Graph) -> None:
    nearest: dict[tuple, tuple] = {}
    distance: dict[tuple, int] = {}
    queue: deque[tuple] = deque()
    for n in graph.nodes:
        if graph.nodes[n][STATE] == NodeState.BURNING:
            nearest[n] = n
            distance[n] = 0
            queue.append(n)
    while queue:
        current = queue.popleft()
        for neighbour in graph.neighbors(current):
            if neighbour not in nearest:
                nearest[neighbour] = nearest[current]
                distance[neighbour] = distance[current] + 1
                queue.append(neighbour)
    graph.graph[_NEAREST_FIRE] = nearest
    graph.graph[_FIRE_DISTANCE] = distance


def distance_to_fire(graph: nx.Graph, node: tuple) -> float:
    dist = graph.graph[_FIRE_DISTANCE].get(node)
    if dist is None:
        return float("inf")
    return dist


def elevation_delta_to_fire(graph: nx.Graph, node: tuple) -> float:
    fire = graph.graph[_NEAREST_FIRE].get(node)
    if fire is None or fire == node:
        return 0.0
    return graph.nodes[node][ELEVATION] - graph.nodes[fire][ELEVATION]


def wind_fire_alignment(graph: nx.Graph, node: tuple) -> float:
    fire = graph.graph[_NEAREST_FIRE].get(node)
    if fire is None or fire == node:
        return 0.0
    ri, ci = node
    fi, fj = fire
    north = fi - ri
    east = ci - fj
    mag = math.sqrt(north**2 + east**2)
    wind_toward_rad = math.radians(graph.graph[WIND_DIRECTION]) + math.pi
    return (north * math.cos(wind_toward_rad) + east * math.sin(wind_toward_rad)) / mag


# ---------------------------------------------------------------------------
# Environment (graph-level)
# ---------------------------------------------------------------------------


def wind_speed(graph: nx.Graph) -> float:
    return graph.graph[WIND_SPEED]


def fuel_moisture(graph: nx.Graph) -> float:
    return graph.graph[FUEL_MOISTURE]


# ---------------------------------------------------------------------------
# Whole-graph state
# ---------------------------------------------------------------------------

TREATMENTS_REMAINING = "treatments_remaining"


def treatments_remaining(graph: nx.Graph) -> int:
    return graph.graph[TREATMENTS_REMAINING]


def total_burning(graph: nx.Graph) -> int:
    return sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNING)


def total_burned(graph: nx.Graph) -> int:
    return sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNED)


def total_unburned(graph: nx.Graph) -> int:
    return sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.UNBURNED)


def total_treated(graph: nx.Graph) -> int:
    return sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.TREATED)
