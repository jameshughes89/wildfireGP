"""Probabilistic wildfire spread model as a cellular automaton on a numpy-backed grid.

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

Vectorisation note
------------------
:func:`spread_step` evaluates all 8 spread directions as whole-array operations rather than iterating burning cells in
Python. For each direction it builds shifted views of the burning and elevation arrays, computes per-cell ignition
probabilities, and accumulates a survival probability per destination cell. A single :func:`Generator.random` draw per
cell decides the outcome.

This is statistically equivalent to the per-source-pair draws the previous implementation made, but not bit-identical:
the same seed produces different specific fire trajectories. The probability that any given cell ignites is unchanged.

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

from wildfireGP.network import GraphState, NodeState, TerrainType

MAX_BURN_STEPS = 5

_C1 = 0.045
_C2 = 0.131
_A_S = 0.078
_KMH_TO_MS = 1.0 / 3.6

_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def spread_step(state: GraphState, rng: np.random.Generator) -> None:
    """Advance the fire simulation by one timestep.

    For each of the 8 spread directions, build a shifted view of the burning/elevation arrays and compute per-cell
    ignition contributions as a whole-array operation. Survival across all directions is accumulated multiplicatively,
    then a single per-cell random draw decides ignition. Burning cells decrement their burn timer and transition to
    BURNED when it reaches zero.
    """
    rows, cols = state.rows, state.cols
    wind_speed_ms = state.wind_speed * _KMH_TO_MS
    wind_toward_rad = math.radians(state.wind_direction) + math.pi
    moisture = state.fuel_moisture

    burning = state.state == NodeState.BURNING
    ignitable = (
        (state.state == NodeState.UNBURNED)
        & (state.terrain != TerrainType.WATER)
        & (state.terrain != TerrainType.ROCK)
        & (state.fuel > 0.0)
    )

    if not burning.any():
        return

    burning_pad = np.pad(burning, 1, constant_values=False)
    elevation_pad = np.pad(state.elevation.astype(np.float32), 1, constant_values=0.0)
    fuel = state.fuel.astype(np.float32)

    survival = np.ones((rows, cols), dtype=np.float32)
    for dr, dc in _DIRECTIONS:
        row_start = 1 - dr
        row_stop = row_start + rows
        col_start = 1 - dc
        col_stop = col_start + cols
        src_burning = burning_pad[row_start:row_stop, col_start:col_stop]
        src_elev = elevation_pad[row_start:row_stop, col_start:col_stop]

        spread_angle = math.atan2(dc, -dr)
        theta = wind_toward_rad - spread_angle
        p_wind = math.exp(_C1 * wind_speed_ms) * math.exp(_C2 * wind_speed_ms * (math.cos(theta) - 1))

        dist_cells = math.sqrt(2) if (dr != 0 and dc != 0) else 1.0
        slope_angle = np.arctan((state.elevation - src_elev) / dist_cells)
        p_slope = np.exp(_A_S * slope_angle)

        p = fuel * (1.0 - moisture) * p_wind * p_slope
        contribution = np.where(src_burning & ignitable, np.clip(p, 0.0, 1.0), 0.0)
        survival *= 1.0 - contribution

    ignition_prob = 1.0 - survival
    newly_ignited = (rng.random((rows, cols)) < ignition_prob) & ignitable

    state.state[newly_ignited] = NodeState.BURNING
    new_timers = np.maximum(1, np.ceil(fuel[newly_ignited] * MAX_BURN_STEPS)).astype(np.int8)
    state.burn_timer[newly_ignited] = new_timers

    state.burn_timer[burning] -= 1
    burned_out = burning & (state.burn_timer <= 0)
    state.state[burned_out] = NodeState.BURNED
    state.fuel[burned_out] = 0.0


def ignition_probability(state: GraphState, src: tuple, dst: tuple) -> float:
    """Compute the ignition probability from BURNING node ``src`` to UNBURNED node ``dst``.

    Returns 0.0 for non-burnable destination terrain. Retained as a single-pair helper for tests and for
    :func:`wildfireGP.render` overlays; the per-step simulation uses the vectorised path in :func:`spread_step`.
    """
    if state.terrain[dst] == TerrainType.WATER or state.terrain[dst] == TerrainType.ROCK:
        return 0.0
    fuel = float(state.fuel[dst])
    if fuel == 0.0:
        return 0.0

    wind_speed_ms = state.wind_speed * _KMH_TO_MS
    wind_toward_rad = math.radians(state.wind_direction) + math.pi

    si, sj = src
    di, dj = dst
    spread_angle = math.atan2(dj - sj, -(di - si))
    theta = wind_toward_rad - spread_angle
    p_wind = math.exp(_C1 * wind_speed_ms) * math.exp(_C2 * wind_speed_ms * (math.cos(theta) - 1))

    elev_diff = float(state.elevation[dst]) - float(state.elevation[src])
    dist_cells = math.sqrt(2) if (si != di and sj != dj) else 1.0
    slope_angle_rad = math.atan(elev_diff / dist_cells)
    p_slope = math.exp(_A_S * slope_angle_rad)

    p = fuel * (1.0 - state.fuel_moisture) * p_wind * p_slope
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p
