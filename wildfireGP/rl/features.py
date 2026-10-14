"""Vectorised per-cell feature extraction for learned (e.g. reinforcement learning) policies.

A learned policy consumes the same 22 features as the GP language (``wildfireGP.language``), in the same slot order, so
the two function classes see identical inputs and the head-to-head comparison stays honest. :data:`FEATURE_ORDER` is
declared literally here rather than derived from the GP primitive set: a drift test (``tests/test_rl_features.py``)
asserts the two agree, which would be tautological if this module imported the order it is meant to check. Declaring
it literally also keeps this module free of a DEAP dependency, so a learned policy needs only numpy and this
input contract.

Why vectorised
--------------
``wildfireGP.features`` exposes one Python function per feature, called once per cell --- the interface a GP tree
needs. Calling those 22 functions across ~2500 cells costs ~13 ms per step, which dominates the ~23 ms simulator
step. Nearly every feature is already backed by a full-grid precompute map, so this module reads those maps directly
and assembles the ``(rows*cols, 22)`` matrix in a handful of array operations instead.

Three features are dict-backed rather than map-backed and are densified here: ``burnable_distance_to_fire`` (from
``state.burnable_fire_distance``) and ``wind_fire_alignment`` / ``elevation_delta_to_fire`` (both from
``state.nearest_fire``). The underlying BFS precomputes in ``wildfireGP.features`` are left untouched --- replacing
them with a vectorised transform would change their tie-breaking and so change GP behaviour, which must stay frozen.

Non-finite values
-----------------
``distance_to_fire`` is ``inf`` once no cell is burning, and ``burnable_distance_to_fire`` is ``inf`` for any cell
with no burnable path to the fire --- up to a third of the grid on barrier landscapes. The GP tolerates this because
``wildfireGP.evaluate._safe_score`` clamps non-finite scores to ``-inf`` after the tree has run. A neural policy has
no such backstop: one ``inf`` entering the first ``Linear`` produces NaN activations and NaN gradients, and training
collapses silently.

Both features are therefore clipped to ``unreachable_distance``, defaulting to ``max(rows, cols)`` --- strictly
greater than the largest achievable chessboard distance on the grid, so "unreachable" stays ordered beyond every
reachable cell. This is a re-encoding of information the GP already sees, not an extra feature: the slot count stays
at 22.
"""

from __future__ import annotations

import math

import numpy as np

from wildfireGP.evaluate import _run_required_precomputes
from wildfireGP.network import GraphState, NodeState

FEATURE_ORDER: tuple[str, ...] = (
    "fuel_level",
    "elevation",
    "slope",
    "mean_neighbour_elevation",
    "mean_neighbour_fuel",
    "burning_neighbour_count",
    "burning_two_hop_count",
    "unburned_neighbour_count",
    "unburnable_neighbour_count",
    "has_treated_neighbour",
    "treated_neighbour_count",
    "distance_to_fire",
    "burnable_distance_to_fire",
    "reachable_unburned_area",
    "wind_fire_alignment",
    "elevation_delta_to_fire",
    "wind_speed",
    "fuel_moisture",
    "total_burning",
    "total_burned",
    "total_unburned",
    "total_treated",
)

N_FEATURES = len(FEATURE_ORDER)

# The precomputes the 22 features need. Excludes betweenness_* and mtt_pathway_count: those are ~100x slower and are
# not in the GP primitive set, so the RL must not see them either.
RL_PRECOMPUTES = frozenset(
    {
        "fire_map",
        "fire_distance_map",
        "burnable_fire_map",
        "reachable_unburned_area",
        "state_counts",
        "neighbourhood_maps",
        "burning_two_hop_map",
    }
)


def run_precomputes(state: GraphState) -> None:
    """Populate the precompute maps the 22 features read. Must be called once per timestep before extraction."""
    _run_required_precomputes(state, RL_PRECOMPUTES)


_INDEX_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _row_col_grids(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _INDEX_CACHE.get((rows, cols))
    if cached is None:
        cached = np.indices((rows, cols), dtype=np.int32)
        _INDEX_CACHE[(rows, cols)] = (cached[0], cached[1])
        return _INDEX_CACHE[(rows, cols)]
    return cached


def default_unreachable_distance(state: GraphState) -> float:
    """Sentinel substituted for infinite distances: strictly beyond any reachable chessboard distance."""
    return float(max(state.rows, state.cols))


def _densify_nearest_fire(state: GraphState) -> tuple[np.ndarray, np.ndarray]:
    """Return (fire_row, fire_col) int32 grids from ``state.nearest_fire``; -1 where no nearest fire is recorded."""
    rows, cols = state.rows, state.cols
    fire_row = np.full((rows, cols), -1, dtype=np.int32)
    fire_col = np.full((rows, cols), -1, dtype=np.int32)
    nearest = state.nearest_fire
    if nearest:
        keys = np.fromiter((v for node in nearest for v in node), dtype=np.int32, count=2 * len(nearest))
        vals = np.fromiter((v for fire in nearest.values() for v in fire), dtype=np.int32, count=2 * len(nearest))
        keys = keys.reshape(-1, 2)
        vals = vals.reshape(-1, 2)
        fire_row[keys[:, 0], keys[:, 1]] = vals[:, 0]
        fire_col[keys[:, 0], keys[:, 1]] = vals[:, 1]
    return fire_row, fire_col


def _densify_burnable_distance(state: GraphState) -> np.ndarray:
    """Return a float64 grid from ``state.burnable_fire_distance``; inf where the cell has no burnable path to fire."""
    rows, cols = state.rows, state.cols
    out = np.full((rows, cols), np.inf, dtype=np.float64)
    distances = state.burnable_fire_distance
    if distances:
        keys = np.fromiter((v for node in distances for v in node), dtype=np.int32, count=2 * len(distances))
        vals = np.fromiter(distances.values(), dtype=np.float64, count=len(distances))
        keys = keys.reshape(-1, 2)
        out[keys[:, 0], keys[:, 1]] = vals
    return out


def extract_features(state: GraphState, unreachable_distance: float | None = None) -> np.ndarray:
    """Return the ``(rows*cols, 22)`` float32 feature matrix in :data:`FEATURE_ORDER`, row-major over cells.

    The caller must have run the precomputes in :data:`RL_PRECOMPUTES` on ``state`` for this timestep --- see
    :func:`run_precomputes`. Features are returned for *every* cell, not only treatable ones; the
    action mask, not the feature matrix, decides eligibility.

    :param state: Landscape state with this step's precomputes already populated.
    :param unreachable_distance: Value substituted for infinite distances. Defaults to ``max(rows, cols)``.
    :return: ``(rows*cols, 22)`` float32 array, guaranteed finite.
    """
    if state.treated_neighbour_count_map is None or state.reachable_unburned_area_map is None:
        raise RuntimeError(
            "Precomputes have not been run for this timestep; call wildfireGP.rl.features.run_precomputes(state) first."
        )
    rows, cols = state.rows, state.cols
    sentinel = default_unreachable_distance(state) if unreachable_distance is None else float(unreachable_distance)
    row_idx, col_idx = _row_col_grids(rows, cols)
    out = np.empty((rows, cols, N_FEATURES), dtype=np.float32)

    out[..., 0] = state.fuel
    out[..., 1] = state.elevation
    out[..., 2] = state.slope
    out[..., 3] = state.mean_neighbour_elevation_map
    out[..., 4] = state.mean_neighbour_fuel_map
    out[..., 5] = state.burning_neighbour_count_map
    out[..., 6] = state.burning_two_hop_count_map
    out[..., 7] = state.unburned_neighbour_count_map
    out[..., 8] = state.unburnable_neighbour_count_map
    out[..., 9] = state.treated_neighbour_count_map > 0
    out[..., 10] = state.treated_neighbour_count_map

    distance = state.fire_distance_map
    if distance is None:
        out[..., 11] = sentinel
    else:
        out[..., 11] = np.where(np.isfinite(distance), distance, sentinel)

    burnable = _densify_burnable_distance(state)
    out[..., 12] = np.where(np.isfinite(burnable), burnable, sentinel)

    out[..., 13] = state.reachable_unburned_area_map

    fire_row, fire_col = _densify_nearest_fire(state)
    # A cell that is itself burning has nearest_fire == itself; wildfireGP returns 0.0 for both fire-relative
    # features in that case, and for cells the BFS never reached.
    has_fire = (fire_row >= 0) & ~((fire_row == row_idx) & (fire_col == col_idx))
    safe_row = np.where(has_fire, fire_row, 0)
    safe_col = np.where(has_fire, fire_col, 0)

    north = (safe_row - row_idx).astype(np.float64)
    east = (col_idx - safe_col).astype(np.float64)
    magnitude = np.sqrt(north**2 + east**2)
    wind_toward_rad = math.radians(state.wind_direction) + math.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        alignment = (north * math.cos(wind_toward_rad) + east * math.sin(wind_toward_rad)) / magnitude
    out[..., 14] = np.where(has_fire, alignment, 0.0)

    elevation = state.elevation.astype(np.float64)
    delta = elevation - elevation[safe_row, safe_col]
    out[..., 15] = np.where(has_fire, delta, 0.0)

    out[..., 16] = state.wind_speed
    out[..., 17] = state.fuel_moisture
    out[..., 18] = state.step_burning
    out[..., 19] = state.step_burned
    out[..., 20] = state.step_unburned
    out[..., 21] = state.step_treated

    return out.reshape(rows * cols, N_FEATURES)


def compute_action_mask(state: GraphState, min_treatment_distance: int = 0) -> np.ndarray:
    """Return the ``(rows*cols,)`` bool mask of treatable cells, matching ``wildfireGP.evaluate._apply_treatments``.

    Eligible cells are UNBURNED with fuel remaining; WATER and ROCK are excluded implicitly since both carry fuel=0.
    A positive ``min_treatment_distance`` additionally excludes cells within that chessboard distance of active fire
    (ground-crew safety zone).
    """
    mask = (state.state == NodeState.UNBURNED) & (state.fuel > 0.0)
    if min_treatment_distance > 0 and state.fire_distance_map is not None:
        mask = mask & (state.fire_distance_map >= min_treatment_distance)
    return mask.reshape(-1)
