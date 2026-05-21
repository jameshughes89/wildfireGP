"""
Per-node and graph-level feature functions for the GP language.

These are the sole building blocks wired into the DEAP primitive set in :mod:`wildfireGP.language` --- every value the
GP language can read about the world comes through one of these functions.

Distance convention
-------------------
Spatial distance is measured in graph hops (Chebyshev distance: max(|Δrow|, |Δcol|)), not Euclidean distance. With a
Moore (8-connectivity) neighbourhood, both cardinal and diagonal neighbours are one spread step away, so hop count
directly represents how many timesteps the fire needs to reach a node.

:func:`wind_fire_alignment` is the one exception: it uses a Euclidean magnitude internally to normalise a direction
vector for the dot product. This is not a distance measurement.
"""

import math
from collections import deque

import numpy as np

from wildfireGP.network import NodeState, GraphState, TerrainType

# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def fuel_level(state: GraphState, node: tuple) -> float:
    return float(state.fuel[node])


def elevation(state: GraphState, node: tuple) -> float:
    return float(state.elevation[node])


def slope(state: GraphState, node: tuple) -> float:
    return float(state.slope[node])


def mean_neighbour_elevation(state: GraphState, node: tuple) -> float:
    neighbours = state.neighbours(node)
    return sum(float(state.elevation[n]) for n in neighbours) / len(neighbours)


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------


def mean_neighbour_fuel(state: GraphState, node: tuple) -> float:
    neighbours = state.neighbours(node)
    return sum(float(state.fuel[n]) for n in neighbours) / len(neighbours)


def burning_neighbour_count(state: GraphState, node: tuple) -> int:
    return sum(1 for n in state.neighbours(node) if state.state[n] == NodeState.BURNING)


def burning_two_hop_count(state: GraphState, node: tuple) -> int:
    one_hop = set(state.neighbours(node))
    two_hop: set[tuple] = set()
    for neighbour in one_hop:
        two_hop.update(state.neighbours(neighbour))
    two_hop.discard(node)
    two_hop.difference_update(one_hop)
    return sum(1 for n in two_hop if state.state[n] == NodeState.BURNING)


def unburned_neighbour_count(state: GraphState, node: tuple) -> int:
    return sum(1 for n in state.neighbours(node) if state.state[n] == NodeState.UNBURNED)


def unburnable_neighbour_count(state: GraphState, node: tuple) -> int:
    count = 0
    for n in state.neighbours(node):
        s = state.state[n]
        t = state.terrain[n]
        if s == NodeState.BURNED or s == NodeState.TREATED or t == TerrainType.WATER or t == TerrainType.ROCK:
            count += 1
    return count


def has_treated_neighbour(state: GraphState, node: tuple) -> float:
    return 1.0 if any(state.state[n] == NodeState.TREATED for n in state.neighbours(node)) else 0.0


def treated_neighbour_count(state: GraphState, node: tuple) -> int:
    return sum(1 for n in state.neighbours(node) if state.state[n] == NodeState.TREATED)


# ---------------------------------------------------------------------------
# Spatial --- requires precompute_fire_map / precompute_burnable_fire_map each simulation step
# ---------------------------------------------------------------------------


def precompute_fire_map(state: GraphState) -> None:
    nearest: dict[tuple, tuple] = {}
    queue: deque[tuple] = deque()
    burning = np.argwhere(state.state == NodeState.BURNING)
    for r, c in burning:
        node = (int(r), int(c))
        nearest[node] = node
        queue.append(node)
    while queue:
        current = queue.popleft()
        for neighbour in state.neighbours(current):
            if neighbour not in nearest:
                nearest[neighbour] = nearest[current]
                queue.append(neighbour)
    state.nearest_fire = nearest


def distance_to_fire(state: GraphState, node: tuple) -> float:
    fire = state.nearest_fire.get(node)
    if fire is None:
        return float("inf")
    return max(abs(node[0] - fire[0]), abs(node[1] - fire[1]))


def precompute_burnable_fire_map(state: GraphState) -> None:
    distance: dict[tuple, int] = {}
    queue: deque[tuple] = deque()
    burning = np.argwhere(state.state == NodeState.BURNING)
    for r, c in burning:
        node = (int(r), int(c))
        distance[node] = 0
        queue.append(node)
    while queue:
        current = queue.popleft()
        for neighbour in state.neighbours(current):
            if neighbour not in distance:
                if state.state[neighbour] == NodeState.UNBURNED and state.terrain[neighbour] == TerrainType.LAND:
                    distance[neighbour] = distance[current] + 1
                    queue.append(neighbour)
    state.burnable_fire_distance = distance


def burnable_distance_to_fire(state: GraphState, node: tuple) -> float:
    dist = state.burnable_fire_distance.get(node)
    if dist is None:
        return float("inf")
    return float(dist)


def precompute_reachable_unburned_area(state: GraphState) -> None:
    """Compute the connected-component size of the unburned-land subgraph for each node.

    All nodes in the same connected component of UNBURNED LAND cells share the same reachable area value, so only one
    BFS per component is needed. Nodes outside the unburned-land subgraph (burning, burned, treated, water, rock) are
    absent from the map and read as 0 via :func:`reachable_unburned_area`.
    """
    area: dict[tuple, int] = {}
    for start in state.nodes():
        if start in area:
            continue
        if state.state[start] != NodeState.UNBURNED or state.terrain[start] != TerrainType.LAND:
            continue
        component: list[tuple] = []
        queue: deque[tuple] = deque([start])
        area[start] = -1
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in state.neighbours(current):
                if neighbour not in area:
                    if state.state[neighbour] == NodeState.UNBURNED and state.terrain[neighbour] == TerrainType.LAND:
                        area[neighbour] = -1
                        queue.append(neighbour)
        size = len(component)
        for n in component:
            area[n] = size
    state.reachable_unburned_area_map = area


def reachable_unburned_area(state: GraphState, node: tuple) -> float:
    return float(state.reachable_unburned_area_map.get(node, 0))


def elevation_delta_to_fire(state: GraphState, node: tuple) -> float:
    fire = state.nearest_fire.get(node)
    if fire is None or fire == node:
        return 0.0
    return float(state.elevation[node]) - float(state.elevation[fire])


def wind_fire_alignment(state: GraphState, node: tuple) -> float:
    fire = state.nearest_fire.get(node)
    if fire is None or fire == node:
        return 0.0
    ri, ci = node
    fi, fj = fire
    north = fi - ri
    east = ci - fj
    mag = math.sqrt(north**2 + east**2)
    wind_toward_rad = math.radians(state.wind_direction) + math.pi
    return (north * math.cos(wind_toward_rad) + east * math.sin(wind_toward_rad)) / mag


# ---------------------------------------------------------------------------
# Environment (graph-level)
# ---------------------------------------------------------------------------


def wind_speed(state: GraphState) -> float:
    return float(state.wind_speed)


def fuel_moisture(state: GraphState) -> float:
    return float(state.fuel_moisture)


# ---------------------------------------------------------------------------
# Whole-graph state --- requires precompute_state_counts each simulation step
# ---------------------------------------------------------------------------


def precompute_state_counts(state: GraphState) -> None:
    """Count cells in each state in a single pass and cache on state."""
    counts = np.bincount(state.state.ravel(), minlength=4)
    state.step_unburned = int(counts[NodeState.UNBURNED])
    state.step_burning = int(counts[NodeState.BURNING])
    state.step_burned = int(counts[NodeState.BURNED])
    state.step_treated = int(counts[NodeState.TREATED])


def total_burning(state: GraphState) -> int:
    return state.step_burning


def total_burned(state: GraphState) -> int:
    return state.step_burned


def total_unburned(state: GraphState) -> int:
    return state.step_unburned


def total_treated(state: GraphState) -> int:
    return state.step_treated
