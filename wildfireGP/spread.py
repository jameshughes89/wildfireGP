"""
Probabilistic wildfire spread model as a cellular automaton on a NetworkX grid graph.

Each timestep, every BURNING node attempts to ignite each UNBURNED neighbour, then decrements its burn timer.
When the timer reaches zero the node transitions to BURNED. WATER, ROCK, and TREATED nodes are non-burnable and are
never ignited.

Alexandridis et al. (2008) use single-timestep burn duration, which is physically reasonable at their 5x5m cell size
where a small vegetation patch is consumed quickly. At a 100x100m cell size a single cell represents a hectare of forest
that in reality burns for hours or days, continuously threatening its neighbours throughout. Collapsing that to one
timestep would make high-fuel and low-fuel cells indistinguishable in burn duration. We therefore use a burn timer
proportional to fuel load: burn_timer = ceil(fuel * MAX_BURN_STEPS) on ignition, so denser fuel burns longer and has
more opportunities to ignite neighbours.

Spread probability
------------------
The ignition probability from a BURNING source node to an UNBURNED destination node is adapted from Alexandridis et al.
(2008), calibrated against the 1990 Spetses Island wildfire:

    p = fuel[dst] * (1 - moisture) * p_wind * p_slope

The original paper uses:
    p_burn = p_h * (1 + p_veg) * (1 + p_den) * p_w * p_s

where p_h is a landscape-specific baseline probability and p_veg, p_den encode categorical vegetation type and density.
We replace these three terms with a single normalised fuel scalar, which serves as a continuous proxy for vegetation
type and density effects.

The paper also omits an explicit moisture term; we include (1 - moisture) as a direct suppression factor consistent with
the Fire Weather Index interpretation of fuel moisture (Van Wagner, 1987).

Wind factor (Alexandridis et al., 2008):
    p_wind = exp(c1 * V) * exp(c2 * V * (cos(theta) - 1))

    V     : wind speed in m/s (graph stores km/h; converted internally)
    theta : angle between wind direction and the src->dst spread direction
    c1    : 0.045  (downwind amplification)
    c2    : 0.131  (directional sensitivity)

Slope factor (Alexandridis et al., 2008):
    p_slope = exp(a_s * phi)

    phi   : slope angle in radians, computed as atan(elev_diff / dist_cells) where elev_diff is the
            normalized [0, 1] elevation difference src->dst and dist_cells is 1.0 cardinal or sqrt(2)
            diagonal. Because elevation is stored normalized rather than in physical metres, phi is
            not the true ground-truth slope angle of any specific landscape; the resulting effect is
            small (<7% per cell at maximum gradient). The functional form matches Alexandridis et al.
            exactly; the magnitude depends on how the synthetic elevation field is interpreted.
    a_s   : 0.078  (Alexandridis et al., 2008)

Scale and timestep
------------------
Alexandridis et al. (2008) use 5x5m cells and a Moore (8-connectivity) neighbourhood. This model matches both. Spotting
(long-range ember transport) is present in the original paper, where skipping several cells corresponds to tens of
meters, occasionally up to ~100m. At 100m resolution the same physical phenomenon falls within a single cell or within
an adjacent cell, and is implicitly captured by the ignition probability, so a separate spotting sub-model is not
warranted. For diagonal neighbours the cell-unit distance used when computing the slope factor is sqrt(2).

The constants c1, c2, and a_s were calibrated at 5x5m resolution and bundle together physical spread rate, cell size,
and timestep duration into a single value. At 100m cells, one timestep implicitly represents a proportionally longer
real-world duration --- the functional form of the ignition probabilities and directional sensitivities remains valid,
but the model makes no claim about the wall-clock duration of a timestep. This is consistent with using the simulator to
rank GP allocation strategies against each other rather than to reproduce physically accurate spread rates.

Wind and moisture must be set on the graph before calling spread_step. Raises KeyError if either is missing.

References
----------
Alexandridis, A., Vakalis, D., Siettos, C.I., & Bafas, G.V. (2008). A cellular automata model for forest fire spread
    prediction: The case of the wildfire that swept through Spetses Island in 1990. Applied Mathematics and
    Computation, 204(1), 191-201. https://doi.org/10.1016/j.amc.2008.06.046
Van Wagner, C.E. (1987). Development and Structure of the Canadian Forest Fire Weather Index System. Canadian Forestry
    Service Technical Report 35.
"""

import math

import networkx as nx
import numpy as np

from wildfireGP.network import (
    BURN_TIMER,
    ELEVATION,
    FUEL,
    FUEL_MOISTURE,
    STATE,
    TERRAIN,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
    TerrainType,
)

MAX_BURN_STEPS = 5

_C1 = 0.045  # Alexandridis et al. (2008): downwind amplification coefficient
_C2 = 0.131  # Alexandridis et al. (2008): directional sensitivity coefficient
_A_S = 0.078  # Alexandridis et al. (2008): slope sensitivity coefficient
_KMH_TO_MS = 1.0 / 3.6


def spread_step(graph: nx.Graph, rng: np.random.Generator) -> None:
    """
    Advance the fire simulation by one timestep.

    Each BURNING node attempts to ignite each UNBURNED neighbour, then decrements its burn timer.
    When the timer reaches zero the node transitions to BURNED.

    :param graph: Landscape graph with WIND_SPEED, WIND_DIRECTION, and FUEL_MOISTURE set as graph-level attributes.
    :param rng: NumPy random generator for reproducible stochastic ignition.
    :raises KeyError: If WIND_SPEED, WIND_DIRECTION, or FUEL_MOISTURE are not set on the graph.
    """
    wind_speed_ms = graph.graph[WIND_SPEED] * _KMH_TO_MS
    wind_dir_rad = math.radians(graph.graph[WIND_DIRECTION])
    moisture = graph.graph[FUEL_MOISTURE]

    to_ignite = []
    to_burn_out = []

    for node in list(graph.nodes):
        if graph.nodes[node][STATE] != NodeState.BURNING:
            continue

        to_ignite.extend(_ignition_targets(graph, node, rng, wind_speed_ms, wind_dir_rad, moisture))
        if _decrement_burn_timer(graph, node):
            to_burn_out.append(node)

    _ignite_nodes(graph, to_ignite)
    _burn_out_nodes(graph, to_burn_out)


def ignition_probability(graph: nx.Graph, src: tuple, dst: tuple) -> float:
    """
    Compute the ignition probability from BURNING node src to UNBURNED node dst.

    :param graph: Landscape graph with WIND_SPEED, WIND_DIRECTION, and FUEL_MOISTURE set.
    :param src: Source (burning) node.
    :param dst: Destination (candidate) node.
    :return: Probability in [0, 1]. Returns 0.0 for non-burnable destination terrain.
    :raises KeyError: If WIND_SPEED, WIND_DIRECTION, or FUEL_MOISTURE are not set on the graph.
    """
    if not _is_burnable(graph, dst):
        return 0.0
    wind_speed_ms = graph.graph[WIND_SPEED] * _KMH_TO_MS
    wind_dir_rad = math.radians(graph.graph[WIND_DIRECTION])
    moisture = graph.graph[FUEL_MOISTURE]
    return _ignition_probability(graph, src, dst, wind_speed_ms, wind_dir_rad, moisture)


def _is_burnable(graph: nx.Graph, node: tuple) -> bool:
    terrain = graph.nodes[node][TERRAIN]
    return terrain not in (TerrainType.WATER, TerrainType.ROCK)


def _can_ignite(graph: nx.Graph, node: tuple) -> bool:
    return graph.nodes[node][STATE] == NodeState.UNBURNED and _is_burnable(graph, node)


def _ignition_targets(
    graph: nx.Graph,
    node: tuple,
    rng: np.random.Generator,
    wind_speed_ms: float,
    wind_dir_rad: float,
    moisture: float,
) -> list[tuple]:
    targets = []
    for neighbour in graph.neighbors(node):
        if not _can_ignite(graph, neighbour):
            continue
        p = _ignition_probability(graph, node, neighbour, wind_speed_ms, wind_dir_rad, moisture)
        if rng.random() < p:
            targets.append(neighbour)
    return targets


def _decrement_burn_timer(graph: nx.Graph, node: tuple) -> bool:
    graph.nodes[node][BURN_TIMER] -= 1
    return graph.nodes[node][BURN_TIMER] <= 0


def _ignite_nodes(graph: nx.Graph, nodes: list[tuple]) -> None:
    for node in nodes:
        if graph.nodes[node][STATE] != NodeState.UNBURNED:
            continue
        graph.nodes[node][STATE] = NodeState.BURNING
        fuel = graph.nodes[node][FUEL]
        graph.nodes[node][BURN_TIMER] = max(1, math.ceil(fuel * MAX_BURN_STEPS))


def _burn_out_nodes(graph: nx.Graph, nodes: list[tuple]) -> None:
    for node in nodes:
        graph.nodes[node][STATE] = NodeState.BURNED
        graph.nodes[node][FUEL] = 0.0


def _ignition_probability(
    graph: nx.Graph,
    src: tuple,
    dst: tuple,
    wind_speed_ms: float,
    wind_dir_rad: float,
    moisture: float,
) -> float:
    fuel = graph.nodes[dst][FUEL]
    if fuel == 0.0:
        return 0.0

    si, sj = src
    di, dj = dst
    spread_angle = math.atan2(dj - sj, -(di - si))  # compass bearing of spread: north=0, east=π/2, south=π

    # wind_dir_rad is FROM direction; convert to TO direction for Alexandridis theta convention
    wind_toward_rad = wind_dir_rad + math.pi
    theta = wind_toward_rad - spread_angle
    p_wind = math.exp(_C1 * wind_speed_ms) * math.exp(_C2 * wind_speed_ms * (math.cos(theta) - 1))

    elev_diff = graph.nodes[dst][ELEVATION] - graph.nodes[src][ELEVATION]
    dist_cells = math.sqrt(2) if (si != di and sj != dj) else 1.0
    slope_angle_rad = math.atan(elev_diff / dist_cells)
    p_slope = math.exp(_A_S * slope_angle_rad)

    p = fuel * (1.0 - moisture) * p_wind * p_slope
    return min(1.0, max(0.0, p))


if __name__ == "__main__":
    from wildfireGP.network import create_grid, set_fuel_moisture, set_wind

    rng = np.random.default_rng(42)
    graph = create_grid(
        50, 50, terrain_smoothing=10, fuel_smoothing=3, water_fraction=0.05, rock_fraction=0.05, seed=42
    )
    set_wind(graph, speed=20.0, direction=45.0)
    set_fuel_moisture(graph, moisture=0.1)

    ignition_node = (25, 25)
    graph.nodes[ignition_node][STATE] = NodeState.BURNING
    graph.nodes[ignition_node][BURN_TIMER] = max(1, math.ceil(graph.nodes[ignition_node][FUEL] * MAX_BURN_STEPS))

    print(f"{'Step':>4}  {'Burning':>8}  {'Burned':>8}  {'Unburned':>9}")
    for step in range(50):
        burning = sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNING)
        burned = sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNED)
        unburned = sum(1 for n in graph.nodes if graph.nodes[n][STATE] == NodeState.UNBURNED)
        print(f"{step:>4}  {burning:>8}  {burned:>8}  {unburned:>9}")
        if burning == 0:
            break
        spread_step(graph, rng)

    print("done")
