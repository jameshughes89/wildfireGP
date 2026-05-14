"""
Per-node and graph-level feature functions for the GP language.

These are the sole building blocks wired into the DEAP primitive set in language.py --- every value the GP language can
read about the world comes through one of these functions, even when the implementation is a direct attribute lookup.

Distance convention
-------------------
Spatial distance is measured in graph hops (Chebyshev distance: max(|Δrow|, |Δcol|)), not Euclidean distance. With a
Moore (8-connectivity) neighbourhood, both cardinal and diagonal neighbours are one spread step away, so hop count
directly represents how many timesteps the fire needs to reach a node. Euclidean distance would be physically accurate
but would conflate proximity in time (what the GP reasons about) with proximity in space.

wind_fire_alignment is the one exception: it uses a Euclidean magnitude internally, but only to normalise a direction
vector for the dot product (cosine similarity always requires a Euclidean norm). This is not a distance measurement.
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


def has_treated_neighbour(graph: nx.Graph, node: tuple) -> float:
    return 1.0 if any(graph.nodes[n][STATE] == NodeState.TREATED for n in graph.neighbors(node)) else 0.0


# ---------------------------------------------------------------------------
# Spatial --- requires precompute_fire_map / precompute_burnable_fire_map each simulation step
# ---------------------------------------------------------------------------

_NEAREST_FIRE = "nearest_fire"
_BURNABLE_FIRE_DISTANCE = "burnable_fire_distance"
_REACHABLE_UNBURNED_AREA = "reachable_unburned_area"


def precompute_fire_map(graph: nx.Graph) -> None:
    nearest: dict[tuple, tuple] = {}
    queue: deque[tuple] = deque()
    for n in graph.nodes:
        if graph.nodes[n][STATE] == NodeState.BURNING:
            nearest[n] = n
            queue.append(n)
    while queue:
        current = queue.popleft()
        for neighbour in graph.neighbors(current):
            if neighbour not in nearest:
                nearest[neighbour] = nearest[current]
                queue.append(neighbour)
    graph.graph[_NEAREST_FIRE] = nearest


def distance_to_fire(graph: nx.Graph, node: tuple) -> float:
    fire = graph.graph[_NEAREST_FIRE].get(node)
    if fire is None:
        return float("inf")
    return max(abs(node[0] - fire[0]), abs(node[1] - fire[1]))


def precompute_burnable_fire_map(graph: nx.Graph) -> None:
    distance: dict[tuple, int] = {}
    queue: deque[tuple] = deque()
    for n in graph.nodes:
        if graph.nodes[n][STATE] == NodeState.BURNING:
            distance[n] = 0
            queue.append(n)
    while queue:
        current = queue.popleft()
        for neighbour in graph.neighbors(current):
            if neighbour not in distance:
                if (
                    graph.nodes[neighbour][STATE] == NodeState.UNBURNED
                    and graph.nodes[neighbour][TERRAIN] == TerrainType.LAND
                ):
                    distance[neighbour] = distance[current] + 1
                    queue.append(neighbour)
    graph.graph[_BURNABLE_FIRE_DISTANCE] = distance


def burnable_distance_to_fire(graph: nx.Graph, node: tuple) -> float:
    dist = graph.graph[_BURNABLE_FIRE_DISTANCE].get(node)
    if dist is None:
        return float("inf")
    return dist


def precompute_reachable_unburned_area(graph: nx.Graph) -> None:
    """Compute the connected-component size of the unburned-land subgraph for each node.

    All nodes in the same connected component of UNBURNED LAND cells share the same
    reachable area value, so only one BFS per component is needed. Nodes outside the
    unburned-land subgraph (burning, burned, treated, water, rock) are assigned 0.
    """
    area: dict[tuple, int] = {}
    for start in graph.nodes:
        if start in area:
            continue
        if graph.nodes[start][STATE] != NodeState.UNBURNED or graph.nodes[start][TERRAIN] != TerrainType.LAND:
            continue
        component: list[tuple] = []
        queue: deque[tuple] = deque([start])
        area[start] = -1  # sentinel: visited but size not yet known
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in graph.neighbors(current):
                if neighbour not in area:
                    if (
                        graph.nodes[neighbour][STATE] == NodeState.UNBURNED
                        and graph.nodes[neighbour][TERRAIN] == TerrainType.LAND
                    ):
                        area[neighbour] = -1
                        queue.append(neighbour)
        size = len(component)
        for n in component:
            area[n] = size
    graph.graph[_REACHABLE_UNBURNED_AREA] = area


def reachable_unburned_area(graph: nx.Graph, node: tuple) -> float:
    return float(graph.graph[_REACHABLE_UNBURNED_AREA].get(node, 0))


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
# Whole-graph state --- requires precompute_state_counts each simulation step
# ---------------------------------------------------------------------------

_STEP_BURNING = "_step_burning"
_STEP_BURNED = "_step_burned"
_STEP_UNBURNED = "_step_unburned"
_STEP_TREATED = "_step_treated"


def precompute_state_counts(graph: nx.Graph) -> None:
    """Count nodes in each state in a single pass and cache on graph.graph."""
    counts = {NodeState.BURNING: 0, NodeState.BURNED: 0, NodeState.UNBURNED: 0, NodeState.TREATED: 0}
    for n in graph.nodes:
        counts[graph.nodes[n][STATE]] += 1
    graph.graph[_STEP_BURNING] = counts[NodeState.BURNING]
    graph.graph[_STEP_BURNED] = counts[NodeState.BURNED]
    graph.graph[_STEP_UNBURNED] = counts[NodeState.UNBURNED]
    graph.graph[_STEP_TREATED] = counts[NodeState.TREATED]


def total_burning(graph: nx.Graph) -> int:
    return graph.graph[_STEP_BURNING]


def total_burned(graph: nx.Graph) -> int:
    return graph.graph[_STEP_BURNED]


def total_unburned(graph: nx.Graph) -> int:
    return graph.graph[_STEP_UNBURNED]


def total_treated(graph: nx.Graph) -> int:
    return graph.graph[_STEP_TREATED]
