"""Wildfire landscape state representation backed by numpy arrays.

A landscape is represented as a :class:`GraphState` dataclass holding per-cell attribute arrays and scalar weather. The
grid is implicitly Moore-connected: each cell has up to 8 neighbours (cardinal + diagonal), matching the Alexandridis et
al. (2008) fire spread model and the eight-connectivity used by all GP features.

Node attributes (each a 2D numpy array of shape (rows, cols))
-------------------------------------------------------------
state : int8
    Dynamic fire state (NodeState). UNBURNED, BURNING, BURNED, or TREATED. Initialised to UNBURNED.
burn_timer : int8
    Remaining timesteps the cell will continue burning. Set to ceil(fuel * MAX_BURN_STEPS) on ignition; decremented each
    step; cell becomes BURNED when timer reaches zero. Initialised to 0.
terrain : int8
    Physical terrain category (TerrainType): LAND, WATER, or ROCK. WATER and ROCK cells have fuel=0 and never ignite.
fuel : float32 in [0, 1]
    Relative fuel load. Higher values increase ignition probability and burn duration. Set to 0 on burnout.
slope : float32 in [0, 1]
    Terrain steepness normalised to [0, 1]. Fire spread rate increases with slope (Rothermel, 1972). Derived from
    numpy.gradient of the elevation heightmap.
elevation : float32 in [0, 1]
    Relative terrain height normalised to [0, 1]. Synthetically generated as the heightmap itself.

Scalar weather (graph-level)
----------------------------
wind_speed       : km/h. Uniform field; per-cell variation is not modelled.
wind_direction   : degrees clockwise from north. 0 = wind from the north.
fuel_moisture    : in [0, 1]. Acts as an ignition gate.
cell_size_m      : side length of each cell in metres. Default 100m.

Excluded attributes
-------------------
Canopy, aspect, spotting, dynamic moisture, particle-level fuel properties — see the previous NetworkX-backed
implementation history for the rationale.

Synthetic data
--------------
:func:`create_grid` generates a synthetic landscape from a Gaussian-smoothed terrain heightmap. Slope is derived from
the terrain gradient. Low-elevation cells become water; high-slope cells become rock (both fuel=0).

References
----------
Rothermel, R.C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research
    Paper INT-115.
Forestry Canada Fire Danger Group. (1992). Development and Structure of the Canadian Forest Fire Behavior Prediction
    System. Information Report ST-X-3.
Van Wagner, C.E. (1987). Development and Structure of the Canadian Forest Fire Weather Index System. Canadian Forestry
    Service Technical Report 35.
Pais, C. et al. (2021). Cell2Fire: A Cell-Based Forest Fire Growth Model to Support Strategic Landscape Management
    Planning. Frontiers in Forests and Global Change. https://doi.org/10.3389/ffgc.2021.692706
"""

import enum
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter


class NodeState(enum.IntEnum):
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    TREATED = 3


class TerrainType(enum.IntEnum):
    LAND = 0
    WATER = 1
    ROCK = 2


@dataclass
class GraphState:
    """Numpy-backed wildfire landscape state.

    Attribute arrays are stored as (rows, cols) numpy arrays. The (row, col) tuple is the canonical node identifier and
    is used everywhere a "node" appears in spread, features, evaluate, and strategies.

    Precompute fields are populated by :mod:`wildfireGP.features`. They are reset implicitly each step by re-invocation.
    """

    rows: int
    cols: int
    state: np.ndarray
    fuel: np.ndarray
    elevation: np.ndarray
    slope: np.ndarray
    terrain: np.ndarray
    burn_timer: np.ndarray
    pristine_fuel: np.ndarray | None = None
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    fuel_moisture: float = 0.0
    cell_size_m: float = 100.0
    nearest_fire: dict = field(default_factory=dict)
    fire_distance_map: np.ndarray | None = None
    burnable_fire_distance: dict = field(default_factory=dict)
    reachable_unburned_area_map: np.ndarray | None = None
    mean_neighbour_elevation_map: np.ndarray | None = None
    mean_neighbour_fuel_map: np.ndarray | None = None
    burning_neighbour_count_map: np.ndarray | None = None
    burning_two_hop_count_map: np.ndarray | None = None
    treated_neighbour_count_map: np.ndarray | None = None
    unburned_neighbour_count_map: np.ndarray | None = None
    unburnable_neighbour_count_map: np.ndarray | None = None
    betweenness_static_map: np.ndarray | None = None
    betweenness_dynamic_map: np.ndarray | None = None
    mtt_pathway_count_map: np.ndarray | None = None
    step_burning: int = 0
    step_burned: int = 0
    step_unburned: int = 0
    step_treated: int = 0

    def copy(self) -> "GraphState":
        """Return a deep copy of the state arrays; precomputes are reset to empty.

        ``pristine_fuel`` is a read-only snapshot of the initial fuel field --- it never changes after creation, so
        the copy shares the underlying array with the original rather than duplicating it.
        """
        return GraphState(
            rows=self.rows,
            cols=self.cols,
            state=self.state.copy(),
            fuel=self.fuel.copy(),
            elevation=self.elevation.copy(),
            slope=self.slope.copy(),
            terrain=self.terrain.copy(),
            burn_timer=self.burn_timer.copy(),
            pristine_fuel=self.pristine_fuel,
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            fuel_moisture=self.fuel_moisture,
            cell_size_m=self.cell_size_m,
        )

    def neighbours(self, node: tuple) -> list[tuple]:
        """Return the Moore (8-connectivity) neighbours of node within grid bounds.

        Backed by a process-wide cache keyed on (rows, cols): the neighbour table is built once per grid shape and
        shared across every SimState of that shape. Callers must not mutate the returned list.
        """
        return _neighbour_table(self.rows, self.cols)[node]

    def nodes(self):
        """Yield every (row, col) in row-major order."""
        for r in range(self.rows):
            for c in range(self.cols):
                yield (r, c)


def create_grid(
    rows: int,
    cols: int,
    terrain_smoothing: float = 3.0,
    fuel_smoothing: float = 3.0,
    water_fraction: float = 0.0,
    rock_fraction: float = 0.0,
    cell_size_m: float = 100.0,
    seed: int | None = None,
) -> GraphState:
    """Create a synthetic landscape GraphState with spatially correlated fuel and slope.

    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :param terrain_smoothing: Gaussian filter sigma for the terrain heightmap.
    :param fuel_smoothing: Gaussian filter sigma for the fuel field.
    :param water_fraction: Fraction of nodes to mark as water (fuel=0), from the lowest-elevation cells.
    :param rock_fraction: Fraction of nodes to mark as rock (fuel=0), from the steepest cells. Rock runs after water and
        can overwrite water cells; the final counts may not exactly match both requested fractions simultaneously.
    :param cell_size_m: Side length of each grid cell in metres. Default 100m.
    :param seed: Random seed for reproducibility.
    :return: A GraphState with all node-attribute arrays populated. Wind and moisture remain at their defaults (0.0).
    """
    rng = np.random.default_rng(seed)

    terrain_height = _normalize(gaussian_filter(rng.random((rows, cols)), sigma=terrain_smoothing))
    dy, dx = np.gradient(terrain_height)
    slope_norm = _normalize(np.sqrt(dx**2 + dy**2))
    fuel_norm = _normalize(gaussian_filter(rng.random((rows, cols)), sigma=fuel_smoothing))

    terrain_type = np.full((rows, cols), TerrainType.LAND, dtype=np.int8)
    if water_fraction > 0.0:
        n_water = max(1, round(rows * cols * water_fraction))
        flat_order = np.argsort(terrain_height.ravel())
        water_mask = np.zeros(rows * cols, dtype=bool)
        water_mask[flat_order[:n_water]] = True
        water_mask = water_mask.reshape(rows, cols)
        fuel_norm[water_mask] = 0.0
        terrain_type[water_mask] = TerrainType.WATER
    if rock_fraction > 0.0:
        n_rock = max(1, round(rows * cols * rock_fraction))
        flat_order = np.argsort(slope_norm.ravel())
        rock_mask = np.zeros(rows * cols, dtype=bool)
        rock_mask[flat_order[-n_rock:]] = True
        rock_mask = rock_mask.reshape(rows, cols)
        fuel_norm[rock_mask] = 0.0
        terrain_type[rock_mask] = TerrainType.ROCK

    fuel_arr = fuel_norm.astype(np.float32)
    return GraphState(
        rows=rows,
        cols=cols,
        state=np.full((rows, cols), NodeState.UNBURNED, dtype=np.int8),
        fuel=fuel_arr,
        elevation=terrain_height.astype(np.float32),
        slope=slope_norm.astype(np.float32),
        terrain=terrain_type,
        burn_timer=np.zeros((rows, cols), dtype=np.int8),
        pristine_fuel=fuel_arr.copy(),
        cell_size_m=cell_size_m,
    )


def set_wind(state: GraphState, speed: float, direction: float) -> None:
    """Set wind speed (km/h) and direction (degrees clockwise from north).

    :raises ValueError: If speed is negative or direction is outside [0, 360).
    """
    if speed < 0.0:
        raise ValueError(f"Wind speed must be >= 0; got {speed}.")
    if direction < 0.0 or direction >= 360.0:
        raise ValueError(f"Wind direction must satisfy 0 <= direction < 360; got {direction}.")
    state.wind_speed = float(speed)
    state.wind_direction = float(direction)


def set_fuel_moisture(state: GraphState, moisture: float) -> None:
    """Set fuel moisture as a fraction in [0, 1].

    :raises ValueError: If moisture is outside [0, 1].
    """
    if moisture < 0.0 or moisture > 1.0:
        raise ValueError(f"Fuel moisture must satisfy 0 <= moisture <= 1; got {moisture}.")
    state.fuel_moisture = float(moisture)


def reset_states(state: GraphState) -> None:
    """Reset all node states to UNBURNED and clear burn timers."""
    state.state[:] = NodeState.UNBURNED
    state.burn_timer[:] = 0


def select_ignition_node(state: GraphState, rng: np.random.Generator, centre_fraction: float = 0.5) -> tuple:
    """Return a randomly selected burnable node from the central region of the landscape.

    Falls back to any burnable node on the full grid if no burnable nodes exist within the central region.

    :param state: Landscape state.
    :param rng: NumPy random generator.
    :param centre_fraction: Fraction of each dimension to use as the candidate region. Must be in (0, 1]. Default 0.5.
    :return: An (row, col) that is UNBURNED, LAND, and fuel > 0.
    :raises ValueError: If no burnable nodes exist anywhere on the grid.
    """
    rows = state.rows
    cols = state.cols
    margin = (1.0 - centre_fraction) / 2.0
    row_lo = int(rows * margin)
    row_hi = int(rows * (1.0 - margin))
    col_lo = int(cols * margin)
    col_hi = int(cols * (1.0 - margin))

    def _valid_mask(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        mask = np.zeros((rows, cols), dtype=bool)
        sub = (
            (state.state[r0:r1, c0:c1] == NodeState.UNBURNED)
            & (state.terrain[r0:r1, c0:c1] == TerrainType.LAND)
            & (state.fuel[r0:r1, c0:c1] > 0.0)
        )
        mask[r0:r1, c0:c1] = sub
        return mask

    mask = _valid_mask(row_lo, row_hi, col_lo, col_hi)
    if not mask.any():
        mask = _valid_mask(0, rows, 0, cols)
    if not mask.any():
        raise ValueError("No burnable nodes available on this landscape.")

    candidates = np.argwhere(mask)
    idx = int(rng.integers(len(candidates)))
    r, c = candidates[idx]
    return (int(r), int(c))


def select_ignition_cluster(state: GraphState, rng: np.random.Generator, size: int = 3) -> list[tuple]:
    """Return a cluster of burnable nodes to ignite simultaneously at t=0.

    Selects one centre via :func:`select_ignition_node`, then expands to up to ``size - 1`` randomly chosen burnable
    LAND neighbours. Multi-node ignition prevents fire self-extinction on the first spread step.
    """
    centre = select_ignition_node(state, rng)
    burnable_neighbours = [
        n
        for n in state.neighbours(centre)
        if state.state[n] == NodeState.UNBURNED and state.terrain[n] == TerrainType.LAND and state.fuel[n] > 0.0
    ]
    rng.shuffle(burnable_neighbours)
    return [centre] + burnable_neighbours[: size - 1]


def _normalize(array: np.ndarray) -> np.ndarray:
    low, hi = array.min(), array.max()
    if hi == low:
        return np.zeros_like(array, dtype=float)
    return (array - low) / (hi - low)


_NEIGHBOUR_TABLE_CACHE: dict[tuple[int, int], dict[tuple[int, int], list[tuple[int, int]]]] = {}


def _neighbour_table(rows: int, cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    table = _NEIGHBOUR_TABLE_CACHE.get((rows, cols))
    if table is not None:
        return table
    table = {}
    for r in range(rows):
        for c in range(cols):
            entry: list[tuple[int, int]] = []
            for dr in (-1, 0, 1):
                nr = r + dr
                if nr < 0 or nr >= rows:
                    continue
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nc = c + dc
                    if 0 <= nc < cols:
                        entry.append((nr, nc))
            table[(r, c)] = entry
    _NEIGHBOUR_TABLE_CACHE[(rows, cols)] = table
    return table
