"""
Probabilistic wildfire spread model as a cellular automaton on a numpy-backed grid.

Each timestep, every BURNING cell attempts to ignite each UNBURNED Moore-neighbour, then decrements its burn timer.
When the timer reaches zero the cell transitions to BURNED. WATER, ROCK, and TREATED cells are non-burnable and never
ignite.

Alexandridis et al. (2008) use single-timestep burn duration, which is physically reasonable at their 5x5m cell size
where a small vegetation patch is consumed quickly. At a 100x100m cell size a single cell represents a hectare of
forest that in reality burns for hours or days, continuously threatening its neighbours throughout. Collapsing that to
one timestep would make high-fuel and low-fuel cells indistinguishable in burn duration. We therefore use a burn timer
proportional to fuel load: burn_timer = ceil(fuel * MAX_BURN_STEPS) on ignition, so denser fuel burns longer and has
more opportunities to ignite neighbours.

Spread probability
------------------
The ignition probability from a BURNING source cell to an UNBURNED destination cell is adapted from Alexandridis et al.
(2008), calibrated against the 1990 Spetses Island wildfire:

    p = fuel[dst] * (1 - moisture) * p_wind * p_slope

The original paper uses:
    p_burn = p_h * (1 + p_veg) * (1 + p_den) * p_w * p_s

where p_h is a landscape-specific baseline probability and p_veg, p_den encode categorical vegetation type and density.
We replace these three terms with a single normalised fuel scalar, which serves as a continuous proxy for vegetation
type and density effects.

The paper also omits an explicit moisture term; we include (1 - moisture) as a direct suppression factor consistent
with the Fire Weather Index interpretation of fuel moisture (Van Wagner, 1987).

Wind factor (Alexandridis et al., 2008):
    p_wind = exp(c1 * V) * exp(c2 * V * (cos(theta) - 1))

    V     : wind speed in m/s (graph stores km/h; converted internally)
    theta : angle between wind direction and the src->dst spread direction
    c1    : 0.045
    c2    : 0.131

Slope factor (Alexandridis et al., 2008):
    p_slope = exp(a_s * phi)

    phi   : slope angle in radians, computed as atan(elev_diff / dist_cells) where elev_diff is the normalized [0, 1]
            elevation difference src->dst and dist_cells is 1.0 cardinal or sqrt(2) diagonal.
    a_s   : 0.078

Scale and timestep
------------------
Alexandridis et al. (2008) use 5x5m cells. The functional form of the ignition probabilities and directional
sensitivities remains valid at coarser resolutions; the model makes no claim about the wall-clock duration of a
timestep.

Wind and moisture must be set on the GraphState before calling :func:`spread_step`.

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

from wildfireGP.network import NodeState, GraphState, TerrainType

MAX_BURN_STEPS = 5

_C1 = 0.045
_C2 = 0.131
_A_S = 0.078
_KMH_TO_MS = 1.0 / 3.6


def spread_step(state: GraphState, rng: np.random.Generator) -> None:
    """
    Advance the fire simulation by one timestep.

    Each BURNING cell attempts to ignite each UNBURNED Moore-neighbour, then decrements its burn timer. When the timer
    reaches zero the cell transitions to BURNED.
    """
    wind_speed_ms = state.wind_speed * _KMH_TO_MS
    wind_dir_rad = math.radians(state.wind_direction)
    moisture = state.fuel_moisture

    to_ignite: list[tuple] = []
    to_burn_out: list[tuple] = []

    burning_nodes = np.argwhere(state.state == NodeState.BURNING)
    for r, c in burning_nodes:
        node = (int(r), int(c))
        to_ignite.extend(_ignition_targets(state, node, rng, wind_speed_ms, wind_dir_rad, moisture))
        if _decrement_burn_timer(state, node):
            to_burn_out.append(node)

    _ignite_nodes(state, to_ignite)
    _burn_out_nodes(state, to_burn_out)


def ignition_probability(state: GraphState, src: tuple, dst: tuple) -> float:
    """
    Compute the ignition probability from BURNING node ``src`` to UNBURNED node ``dst``.

    Returns 0.0 for non-burnable destination terrain.
    """
    if not _is_burnable(state, dst):
        return 0.0
    wind_speed_ms = state.wind_speed * _KMH_TO_MS
    wind_dir_rad = math.radians(state.wind_direction)
    moisture = state.fuel_moisture
    return _ignition_probability(state, src, dst, wind_speed_ms, wind_dir_rad, moisture)


def _is_burnable(state: GraphState, node: tuple) -> bool:
    terrain = state.terrain[node]
    return terrain != TerrainType.WATER and terrain != TerrainType.ROCK


def _can_ignite(state: GraphState, node: tuple) -> bool:
    return state.state[node] == NodeState.UNBURNED and _is_burnable(state, node)


def _ignition_targets(
    state: GraphState,
    node: tuple,
    rng: np.random.Generator,
    wind_speed_ms: float,
    wind_dir_rad: float,
    moisture: float,
) -> list[tuple]:
    targets: list[tuple] = []
    for neighbour in state.neighbours(node):
        if not _can_ignite(state, neighbour):
            continue
        p = _ignition_probability(state, node, neighbour, wind_speed_ms, wind_dir_rad, moisture)
        if rng.random() < p:
            targets.append(neighbour)
    return targets


def _decrement_burn_timer(state: GraphState, node: tuple) -> bool:
    state.burn_timer[node] -= 1
    return state.burn_timer[node] <= 0


def _ignite_nodes(state: GraphState, nodes: list[tuple]) -> None:
    for node in nodes:
        if state.state[node] != NodeState.UNBURNED:
            continue
        state.state[node] = NodeState.BURNING
        fuel = float(state.fuel[node])
        state.burn_timer[node] = max(1, math.ceil(fuel * MAX_BURN_STEPS))


def _burn_out_nodes(state: GraphState, nodes: list[tuple]) -> None:
    for node in nodes:
        state.state[node] = NodeState.BURNED
        state.fuel[node] = 0.0


def _ignition_probability(
    state: GraphState,
    src: tuple,
    dst: tuple,
    wind_speed_ms: float,
    wind_dir_rad: float,
    moisture: float,
) -> float:
    fuel = float(state.fuel[dst])
    if fuel == 0.0:
        return 0.0

    si, sj = src
    di, dj = dst
    spread_angle = math.atan2(dj - sj, -(di - si))

    wind_toward_rad = wind_dir_rad + math.pi
    theta = wind_toward_rad - spread_angle
    p_wind = math.exp(_C1 * wind_speed_ms) * math.exp(_C2 * wind_speed_ms * (math.cos(theta) - 1))

    elev_diff = float(state.elevation[dst]) - float(state.elevation[src])
    dist_cells = math.sqrt(2) if (si != di and sj != dj) else 1.0
    slope_angle_rad = math.atan(elev_diff / dist_cells)
    p_slope = math.exp(_A_S * slope_angle_rad)

    p = fuel * (1.0 - moisture) * p_wind * p_slope
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p
