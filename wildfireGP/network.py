"""
Wildfire landscape graph construction and management.

This module defines the graph contract used throughout wildfireGP. A landscape is represented as a NetworkX grid graph
where each node corresponds to a spatial patch and carries three attributes: state, fuel, and slope. Fire weather (wind
and moisture) is stored as graph-level attributes, as these quantities are meteorologically driven and effectively
uniform across the landscape at the scales we model.

Node attributes
---------------
state : NodeState
    Dynamic fire state of the patch (UNBURNED, BURNING, BURNED, or TREATED). Initialised to UNBURNED; modified by the
    spread simulation.
fuel : float in [0, 1]
    Relative fuel load. Higher values indicate more burnable material and increase ignition probability and burn
    duration. Currently assigned via spatially correlated synthetic generation. When real data loading is added, this
    will be derived from categorical fuel type classifications (Canadian FBP system or US LANDFIRE Scott-Burgan models).
slope : float in [0, 1]
    Terrain steepness normalised to [0, 1]. Fire spread rate increases with slope (Rothermel, 1972). Currently
    generated synthetically; will be derived from a DEM via numpy.gradient when real data loading is added.

Graph-level attributes
----------------------
wind_speed : float
    Wind speed in km/h. Wind is the dominant driver of fire spread direction and rate (Rothermel, 1972).
    Modelled as a uniform field — per-node variation would require meteorological downscaling data not generally
    available at the resolution we operate at.
wind_direction : float
    Wind direction in degrees (0 = north, clockwise). Uniform across the landscape.
fuel_moisture : float in [0, 1]
    Relative fuel moisture. Acts as an ignition gate: high moisture suppresses spread, low moisture amplifies
    it. In practice this corresponds to the Fine Fuel Moisture Code (FFMC) from the Canadian Fire Weather Index
    system (Van Wagner, 1987).

Excluded attributes
-------------------
The following were considered and deliberately omitted:

- Canopy attributes (height, bulk density, base height, cover): relevant for crown fire modelling. We model
  surface fire only, which is appropriate for the GP resource allocation framing. Adding canopy attributes
  would require 3+ additional node features with marginal benefit at this scale.
- Aspect: compass direction a slope faces. A second-order microclimate effect dominated by wind direction and
  slope magnitude at the scales we model.
- Spotting (ember transport): long-range stochastic ignition ahead of the fire front. Requires a separate
  sub-model and is out of scope for a graph-based CA.
- Dynamic fuel moisture: temporal variation in moisture during a single run. Held constant per run, consistent
  with standard probabilistic CA practice.
- Particle-level fuel properties (heat content, particle size, mineral content): Rothermel uses 13 fuel
  descriptors; categorical fuel type captures sufficient variance for our purposes.

Synthetic data
--------------
create_grid() generates a synthetic landscape using spatially correlated random fields (Gaussian-smoothed
noise) for both fuel and slope. The smoothing parameter controls patch size, producing landscapes with
realistic spatial structure rather than independent per-node noise.

References
----------
Rothermel, R.C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest
    Service Research Paper INT-115.
Forestry Canada Fire Danger Group. (1992). Development and Structure of the Canadian Forest Fire Behavior
    Prediction System. Information Report ST-X-3.
Van Wagner, C.E. (1987). Development and Structure of the Canadian Forest Fire Weather Index System.
    Canadian Forestry Service Technical Report 35.
Pais, C. et al. (2021). Cell2Fire: A Cell-Based Forest Fire Growth Model to Support Strategic Landscape
    Management Planning. Frontiers in Forests and Global Change. https://doi.org/10.3389/ffgc.2021.692706
"""

import enum

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Attribute name constants
# ---------------------------------------------------------------------------

STATE = "state"
FUEL = "fuel"
SLOPE = "slope"
WIND_SPEED = "wind_speed"
WIND_DIRECTION = "wind_direction"
FUEL_MOISTURE = "fuel_moisture"


# ---------------------------------------------------------------------------
# Node state
# ---------------------------------------------------------------------------


class NodeState(enum.Enum):
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    TREATED = 3


# ---------------------------------------------------------------------------
# Synthetic graph construction
# ---------------------------------------------------------------------------


def create_grid(rows: int, cols: int, smoothing: float = 3.0, seed: int | None = None) -> nx.Graph:
    """Create a synthetic landscape grid graph with spatially correlated fuel and slope.

    Gaussian-smoothed random arrays produce realistic patchy fuel distributions and gradual terrain rather
    than independent per-node noise.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    smoothing : float
        Gaussian filter sigma. Higher values produce smoother, larger-scale spatial patterns. Default 3.0
        gives patch sizes of roughly 3 nodes.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Grid graph with STATE, FUEL, and SLOPE node attributes. Wind and moisture are not set; call
        set_wind() and set_fuel_moisture() before running simulations.
    """
    rng = np.random.default_rng(seed)
    graph = nx.grid_2d_graph(rows, cols)

    fuel_norm = _normalize(gaussian_filter(rng.random((rows, cols)), sigma=smoothing))
    slope_norm = _normalize(gaussian_filter(rng.random((rows, cols)), sigma=smoothing))

    _attach_node_attributes(graph, fuel_norm, slope_norm)
    return graph


# ---------------------------------------------------------------------------
# Graph-level attribute helpers
# ---------------------------------------------------------------------------


def set_wind(graph: nx.Graph, speed: float, direction: float) -> None:
    """Set wind speed and direction as graph-level attributes.

    Parameters
    ----------
    speed : float
        Wind speed in km/h.
    direction : float
        Wind direction in degrees (0 = north, clockwise).
    """
    graph.graph[WIND_SPEED] = speed
    graph.graph[WIND_DIRECTION] = direction


def set_fuel_moisture(graph: nx.Graph, moisture: float) -> None:
    """Set fuel moisture as a graph-level attribute.

    Parameters
    ----------
    moisture : float
        Relative fuel moisture in [0, 1]. 0 = bone dry, 1 = saturated.
    """
    graph.graph[FUEL_MOISTURE] = moisture


# ---------------------------------------------------------------------------
# Simulation utilities
# ---------------------------------------------------------------------------


def reset_states(graph: nx.Graph) -> None:
    """Reset all node states to UNBURNED.

    Call between GP fitness evaluations to reuse the same graph without rebuilding it.
    """
    for node in graph.nodes:
        graph.nodes[node][STATE] = NodeState.UNBURNED


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _attach_node_attributes(graph: nx.Graph, fuel_array: np.ndarray, slope_array: np.ndarray) -> None:
    for i, j in graph.nodes:
        graph.nodes[(i, j)][STATE] = NodeState.UNBURNED
        graph.nodes[(i, j)][FUEL] = float(fuel_array[i, j])
        graph.nodes[(i, j)][SLOPE] = float(slope_array[i, j])


def _normalize(array: np.ndarray) -> np.ndarray:
    lo, hi = array.min(), array.max()
    if hi == lo:
        return np.zeros_like(array, dtype=float)
    return (array - lo) / (hi - lo)
