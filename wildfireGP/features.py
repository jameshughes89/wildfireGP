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
from scipy.ndimage import label

from wildfireGP.network import GraphState, NodeState, TerrainType

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
    if state.mean_neighbour_elevation_map is not None:
        return float(state.mean_neighbour_elevation_map[node])
    neighbours = state.neighbours(node)
    return sum(float(state.elevation[n]) for n in neighbours) / len(neighbours)


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------


def mean_neighbour_fuel(state: GraphState, node: tuple) -> float:
    if state.mean_neighbour_fuel_map is not None:
        return float(state.mean_neighbour_fuel_map[node])
    neighbours = state.neighbours(node)
    return sum(float(state.fuel[n]) for n in neighbours) / len(neighbours)


def burning_neighbour_count(state: GraphState, node: tuple) -> int:
    if state.burning_neighbour_count_map is not None:
        return int(state.burning_neighbour_count_map[node])
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
    if state.unburned_neighbour_count_map is not None:
        return int(state.unburned_neighbour_count_map[node])
    return sum(1 for n in state.neighbours(node) if state.state[n] == NodeState.UNBURNED)


def unburnable_neighbour_count(state: GraphState, node: tuple) -> int:
    if state.unburnable_neighbour_count_map is not None:
        return int(state.unburnable_neighbour_count_map[node])
    count = 0
    for n in state.neighbours(node):
        s = state.state[n]
        t = state.terrain[n]
        if s == NodeState.BURNED or s == NodeState.TREATED or t == TerrainType.WATER or t == TerrainType.ROCK:
            count += 1
    return count


def has_treated_neighbour(state: GraphState, node: tuple) -> float:
    if state.treated_neighbour_count_map is not None:
        return 1.0 if state.treated_neighbour_count_map[node] > 0 else 0.0
    return 1.0 if any(state.state[n] == NodeState.TREATED for n in state.neighbours(node)) else 0.0


def treated_neighbour_count(state: GraphState, node: tuple) -> int:
    if state.treated_neighbour_count_map is not None:
        return int(state.treated_neighbour_count_map[node])
    return sum(1 for n in state.neighbours(node) if state.state[n] == NodeState.TREATED)


def precompute_neighbourhood_maps(state: GraphState) -> None:
    neighbour_count = _neighbour_count_map(state.rows, state.cols)
    state.mean_neighbour_elevation_map = _moore_sum(state.elevation) / neighbour_count
    state.mean_neighbour_fuel_map = _moore_sum(state.fuel) / neighbour_count
    state.burning_neighbour_count_map = _moore_sum((state.state == NodeState.BURNING).astype(np.int16))
    state.treated_neighbour_count_map = _moore_sum((state.state == NodeState.TREATED).astype(np.int16))
    state.unburned_neighbour_count_map = _moore_sum((state.state == NodeState.UNBURNED).astype(np.int16))
    unburnable_mask = (
        (state.state == NodeState.BURNED)
        | (state.state == NodeState.TREATED)
        | (state.terrain == TerrainType.WATER)
        | (state.terrain == TerrainType.ROCK)
    )
    state.unburnable_neighbour_count_map = _moore_sum(unburnable_mask.astype(np.int16))


def update_neighbourhood_maps_after_treatment(state: GraphState, node: tuple) -> None:
    neighbours = state.neighbours(node)
    if state.treated_neighbour_count_map is not None:
        for neighbour in neighbours:
            state.treated_neighbour_count_map[neighbour] += 1
    if state.unburned_neighbour_count_map is not None:
        for neighbour in neighbours:
            state.unburned_neighbour_count_map[neighbour] -= 1
    if state.unburnable_neighbour_count_map is not None:
        for neighbour in neighbours:
            state.unburnable_neighbour_count_map[neighbour] += 1


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
    """Compute the connected-component size of the unburned-land subgraph for each node."""
    unburned_land = (state.state == NodeState.UNBURNED) & (state.terrain == TerrainType.LAND)
    labels, _ = label(unburned_land, structure=np.ones((3, 3), dtype=np.int8))
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    state.reachable_unburned_area_map = component_sizes[labels]


def reachable_unburned_area(state: GraphState, node: tuple) -> float:
    if state.reachable_unburned_area_map is None:
        return 0.0
    return float(state.reachable_unburned_area_map[node])


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


_NEIGHBOUR_COUNT_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _moore_sum(array: np.ndarray) -> np.ndarray:
    padded = np.pad(array, 1)
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )


def _neighbour_count_map(rows: int, cols: int) -> np.ndarray:
    cached = _NEIGHBOUR_COUNT_CACHE.get((rows, cols))
    if cached is not None:
        return cached
    counts = _moore_sum(np.ones((rows, cols), dtype=np.int16)).astype(np.float32)
    _NEIGHBOUR_COUNT_CACHE[(rows, cols)] = counts
    return counts
