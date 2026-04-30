"""
Probabilistic wildfire spread model as a cellular automaton on a NetworkX grid graph.

Each timestep, every BURNING node attempts to ignite each UNBURNED neighbour, then immediately transitions to BURNED.
WATER, ROCK, and TREATED nodes are non-burnable and are never ignited. This matches the single-timestep burn duration
used by Alexandridis et al. (2008).

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
    p_slope = exp(a_s * tan(phi))

    phi   : slope angle between src and dst, derived from elevation difference and cell size
    a_s   : 0.078  (slope sensitivity)

Scale and timestep
------------------
Alexandridis et al. (2008) use 5x5m cells. This model targets 100x100m cells, representing a 25 km² landscape at
50x50 nodes — consistent with initial attack resource allocation decisions. Spotting (long-range ember transport) is
present in the original paper, where skipping 3-4 cells represents a realistic 15-20m ember carry. At 100m resolution
the same physical phenomenon falls within a single cell and is implicitly captured by the ignition probability, so a
separate spotting sub-model is not warranted.

The constants c1, c2, and a_s were calibrated at 5x5m resolution and bundle together physical spread rate, cell size,
and timestep duration into a single value. At 100m cells, one timestep implicitly represents a proportionally longer
real-world duration — the ignition probabilities and directional sensitivities remain valid, but the model makes no
claim about the wall-clock duration of a timestep. This is consistent with using the simulator to rank GP allocation
strategies against each other rather than to reproduce physically accurate spread rates.

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

import numpy as np

from wildfireGP.network import (
    CELL_SIZE,
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

_C1 = 0.045   # Alexandridis et al. (2008): downwind amplification coefficient
_C2 = 0.131   # Alexandridis et al. (2008): directional sensitivity coefficient
_A_S = 0.078  # Alexandridis et al. (2008): slope sensitivity coefficient
_KMH_TO_MS = 1.0 / 3.6


def spread_step(graph, rng: np.random.Generator) -> None:
    """
    Advance the fire simulation by one timestep.

    Each BURNING node attempts to ignite each UNBURNED neighbour, then transitions to BURNED.

    :param graph: Landscape graph with WIND_SPEED, WIND_DIRECTION, and FUEL_MOISTURE set as graph-level attributes.
    :param rng: NumPy random generator for reproducible stochastic ignition.
    :raises KeyError: If WIND_SPEED, WIND_DIRECTION, or FUEL_MOISTURE are not set on the graph.
    """
    wind_speed_ms = graph.graph[WIND_SPEED] * _KMH_TO_MS
    wind_dir_rad = math.radians(graph.graph[WIND_DIRECTION])
    moisture = graph.graph[FUEL_MOISTURE]
    cell_size = graph.graph[CELL_SIZE]

    to_ignite = []
    to_burn_out = []

    for node in list(graph.nodes):
        if graph.nodes[node][STATE] != NodeState.BURNING:
            continue

        for neighbour in graph.neighbors(node):
            if graph.nodes[neighbour][STATE] != NodeState.UNBURNED:
                continue
            if not _is_burnable(graph, neighbour):
                continue
            p = _ignition_probability(graph, node, neighbour, wind_speed_ms, wind_dir_rad, moisture, cell_size)
            if rng.random() < p:
                to_ignite.append(neighbour)

        to_burn_out.append(node)

    for node in to_ignite:
        if graph.nodes[node][STATE] == NodeState.UNBURNED:
            graph.nodes[node][STATE] = NodeState.BURNING

    for node in to_burn_out:
        graph.nodes[node][STATE] = NodeState.BURNED


def ignition_probability(graph, src: tuple, dst: tuple) -> float:
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
    cell_size = graph.graph[CELL_SIZE]
    return _ignition_probability(graph, src, dst, wind_speed_ms, wind_dir_rad, moisture, cell_size)


def _is_burnable(graph, node: tuple) -> bool:
    terrain = graph.nodes[node][TERRAIN]
    return terrain not in (TerrainType.WATER, TerrainType.ROCK)


def _ignition_probability(graph, src: tuple, dst: tuple, wind_speed_ms: float, wind_dir_rad: float,
                           moisture: float, cell_size: float) -> float:
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
    slope_tan = elev_diff / cell_size if cell_size > 0 else 0.0
    p_slope = math.exp(_A_S * slope_tan)

    p = fuel * (1.0 - moisture) * p_wind * p_slope
    return min(1.0, max(0.0, p))
