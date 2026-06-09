"""Per-node and graph-level feature functions for the GP language.

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

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.ndimage import convolve, distance_transform_cdt, label

from wildfireGP.network import GraphState, NodeState, TerrainType
from wildfireGP.spread import _A_S, _C1, _C2, _DIRECTIONS, _KMH_TO_MS

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
    if state.burning_two_hop_count_map is not None:
        return int(state.burning_two_hop_count_map[node])
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


def precompute_burning_two_hop_map(state: GraphState) -> None:
    burning = (state.state == NodeState.BURNING).astype(np.int16)
    state.burning_two_hop_count_map = convolve(burning, _TWO_HOP_RING_KERNEL, mode="constant", cval=0)


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
# Spatial --- requires precompute_fire_map / precompute_fire_distance_map /
# precompute_burnable_fire_map each simulation step
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
    precompute_fire_distance_map(state)


def precompute_fire_distance_map(state: GraphState) -> None:
    burning = state.state == NodeState.BURNING
    if not burning.any():
        state.fire_distance_map = np.full(state.state.shape, np.inf, dtype=np.float32)
        return
    state.fire_distance_map = distance_transform_cdt(~burning, metric="chessboard").astype(np.float32)


def distance_to_fire(state: GraphState, node: tuple) -> float:
    if state.fire_distance_map is None:
        return float("inf")
    return float(state.fire_distance_map[node])


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


# ---------------------------------------------------------------------------
# Graph-theoretic --- precompute_betweenness_static / precompute_betweenness_dynamic /
# precompute_mtt_pathway_count. Built on a NetworkX view of the landscape with
# edge weights = inverse mean fuel (proxy for resistance to spread).
# ---------------------------------------------------------------------------

_EDGE_WEIGHT_EPS = 0.01
_BETWEENNESS_SAMPLE_K = 100
_BETWEENNESS_SEED = 0


def _edge_weight(state: GraphState, u: tuple, v: tuple) -> float:
    return 1.0 / (0.5 * (float(state.fuel[u]) + float(state.fuel[v])) + _EDGE_WEIGHT_EPS)


def _build_landscape_graph(state: GraphState, node_mask: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    nodes = [(int(r), int(c)) for r, c in np.argwhere(node_mask)]
    g.add_nodes_from(nodes)
    for u in nodes:
        for v in state.neighbours(u):
            if g.has_node(v) and v > u:
                g.add_edge(u, v, weight=_edge_weight(state, u, v))
    return g


def precompute_betweenness_static(state: GraphState) -> None:
    """Compute betweenness centrality on the full burnable-LAND subgraph once and cache it.

    Idempotent: subsequent calls within the same evaluation are no-ops. Edges are weighted by the inverse mean fuel of
    their endpoints, so high-fuel corridors carry low weight (easy for fire to traverse). Matches the offline pre-fire
    treatment-planning use case from Pais et al. (2021).

    Uses Brandes & Pich (2007) k-source sampling (k=:data:`_BETWEENNESS_SAMPLE_K`) for tractable runtime on 50x50
    grids; exact BC takes ~37 s/sim vs ~1.6 s/sim approximate. Rank ordering is preserved within ~3% across seeds.
    """
    if state.betweenness_static_map is not None:
        return
    arr = np.zeros((state.rows, state.cols), dtype=np.float32)
    g = _build_landscape_graph(state, state.terrain == TerrainType.LAND)
    if len(g) > 0:
        k = min(_BETWEENNESS_SAMPLE_K, len(g))
        bc = nx.betweenness_centrality(g, k=k, weight="weight", normalized=True, seed=_BETWEENNESS_SEED)
        for node, score in bc.items():
            arr[node] = score
    state.betweenness_static_map = arr


def precompute_betweenness_dynamic(state: GraphState) -> None:
    """Compute betweenness centrality on the current UNBURNED-LAND subgraph each timestep.

    BURNED, BURNING, and TREATED cells are removed from the graph before centrality is recomputed, so the
    high-centrality cells shift as the fire grows.

    Uses Brandes & Pich (2007) k-source sampling (k=:data:`_BETWEENNESS_SAMPLE_K`) for tractable runtime; exact BC
    is ~600 s/sim vs ~37 s/sim approximate on 50x50 grids.
    """
    arr = np.zeros((state.rows, state.cols), dtype=np.float32)
    mask = (state.terrain == TerrainType.LAND) & (state.state == NodeState.UNBURNED)
    g = _build_landscape_graph(state, mask)
    if len(g) > 0:
        k = min(_BETWEENNESS_SAMPLE_K, len(g))
        bc = nx.betweenness_centrality(g, k=k, weight="weight", normalized=True, seed=_BETWEENNESS_SEED)
        for node, score in bc.items():
            arr[node] = score
    state.betweenness_dynamic_map = arr


def _build_directed_spread_graph(state: GraphState) -> scipy.sparse.csr_matrix:
    """Build the directed spread graph as a sparse adjacency matrix.

    For each directed Moore-edge u -> v between two active cells, weight = -log(P(ignite u -> v)) using the exact
    Alexandridis (2008) per-edge ignition probability from :func:`wildfireGP.spread.spread_step`. Includes the wind- and
    slope-direction asymmetries: P(u -> v) and P(v -> u) are different whenever wind speed > 0 or there is any
    elevation difference between u and v.

    Active = LAND, not BURNED, not TREATED, fuel > 0. Cells outside this mask have no incoming or outgoing edges.
    """
    rows, cols = state.rows, state.cols
    n = rows * cols
    active = (
        (state.terrain == TerrainType.LAND)
        & (state.state != NodeState.BURNED)
        & (state.state != NodeState.TREATED)
        & (state.fuel > 0)
    )
    if not active.any():
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    wind_speed_ms = float(state.wind_speed) * _KMH_TO_MS
    wind_toward_rad = math.radians(float(state.wind_direction)) + math.pi
    moisture = float(state.fuel_moisture)
    fuel = state.fuel.astype(np.float64)
    elevation = state.elevation.astype(np.float64)

    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    p_floor = 1e-30

    for dr, dc in _DIRECTIONS:
        spread_angle = math.atan2(dc, -dr)
        theta = wind_toward_rad - spread_angle
        p_wind = math.exp(_C1 * wind_speed_ms) * math.exp(_C2 * wind_speed_ms * (math.cos(theta) - 1))
        dist_cells = math.sqrt(2) if (dr != 0 and dc != 0) else 1.0

        r_lo, r_hi = max(0, -dr), rows - max(0, dr)
        c_lo, c_hi = max(0, -dc), cols - max(0, dc)
        if r_lo >= r_hi or c_lo >= c_hi:
            continue

        src_slice = (slice(r_lo, r_hi), slice(c_lo, c_hi))
        dst_slice = (slice(r_lo + dr, r_hi + dr), slice(c_lo + dc, c_hi + dc))
        valid = active[src_slice] & active[dst_slice]
        if not valid.any():
            continue

        elev_diff = elevation[dst_slice] - elevation[src_slice]
        slope_angle = np.arctan(elev_diff / dist_cells)
        p_slope = np.exp(_A_S * slope_angle)
        p = fuel[dst_slice] * (1.0 - moisture) * p_wind * p_slope
        p = np.clip(p, p_floor, 1.0)
        w = -np.log(p)

        r_grid, c_grid = np.meshgrid(np.arange(r_lo, r_hi), np.arange(c_lo, c_hi), indexing="ij")
        src_flat = r_grid * cols + c_grid
        dst_flat = (r_grid + dr) * cols + (c_grid + dc)
        flat_mask = valid.ravel()
        src_list.append(src_flat.ravel()[flat_mask])
        dst_list.append(dst_flat.ravel()[flat_mask])
        w_list.append(w.ravel()[flat_mask])

    if not src_list:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)
    src_idx = np.concatenate(src_list)
    dst_idx = np.concatenate(dst_list)
    weights = np.concatenate(w_list)
    return scipy.sparse.csr_matrix((weights, (src_idx, dst_idx)), shape=(n, n))


def precompute_mtt_pathway_count(state: GraphState) -> None:
    """Count minimum-travel-time fire-arrival pathways through each cell.

    Build a directed graph whose edge weights are derived from the simulator's own per-edge ignition probability
    (Alexandridis 2008 -- fuel, moisture, wind speed, wind direction, slope and slope direction). The edge weight for
    u -> v is -log(P(ignite u -> v)), so a shortest path from the burning set to any cell is the most-likely fire
    propagation route. For every perimeter LAND cell reachable from the burning set, every node along its single
    shortest path is credited one count.

    Compared to the earlier networkx-undirected fuel-only formulation, this version captures the directional
    asymmetries from wind and slope and stays consistent with the simulator's own physics. Computed via scipy
    sparse Dijkstra.

    Reference: Finney (2001) MTT method (PNW-GTR-610), adapted to use the simulator's own ignition probabilities.
    """
    rows, cols = state.rows, state.cols
    counts = np.zeros((rows, cols), dtype=np.float32)

    burning_mask = state.state == NodeState.BURNING
    if not burning_mask.any():
        state.mtt_pathway_count_map = counts
        return

    graph = _build_directed_spread_graph(state)
    if graph.nnz == 0:
        state.mtt_pathway_count_map = counts
        return

    burnable = (
        (state.terrain == TerrainType.LAND)
        & (state.state != NodeState.BURNED)
        & (state.state != NodeState.TREATED)
        & (state.fuel > 0)
    )
    perim = np.zeros((rows, cols), dtype=bool)
    perim[0, :] = True
    perim[-1, :] = True
    perim[:, 0] = True
    perim[:, -1] = True
    perim &= burnable
    boundary_indices = np.flatnonzero(perim.ravel())
    if len(boundary_indices) == 0:
        state.mtt_pathway_count_map = counts
        return

    burning_indices = np.flatnonzero(burning_mask.ravel())
    distances, predecessors, _sources = scipy.sparse.csgraph.dijkstra(
        graph,
        indices=burning_indices,
        return_predecessors=True,
        directed=True,
        min_only=True,
    )

    for target_flat in boundary_indices:
        if not np.isfinite(distances[target_flat]):
            continue
        cur = int(target_flat)
        while cur >= 0:
            counts[cur // cols, cur % cols] += 1
            cur = int(predecessors[cur])

    state.mtt_pathway_count_map = counts


def betweenness_static(state: GraphState, node: tuple) -> float:
    if state.betweenness_static_map is None:
        return 0.0
    return float(state.betweenness_static_map[node])


def betweenness_dynamic(state: GraphState, node: tuple) -> float:
    if state.betweenness_dynamic_map is None:
        return 0.0
    return float(state.betweenness_dynamic_map[node])


def mtt_pathway_count(state: GraphState, node: tuple) -> float:
    if state.mtt_pathway_count_map is None:
        return 0.0
    return float(state.mtt_pathway_count_map[node])


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
_TWO_HOP_RING_KERNEL = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.int16,
)


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
