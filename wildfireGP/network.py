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
    Terrain steepness normalised to [0, 1]. Fire spread rate increases with slope (Rothermel, 1972). Derived from a
    synthetic terrain heightmap via numpy.gradient; will use a real DEM when real data loading is added.

Graph-level attributes
----------------------
cell_size_m : float
    Side length of each grid cell in metres. Defaults to 100m, matching Canadian FBP operational scale. A 50x50 grid
    at 100m represents a 5km x 5km landscape, which is a meaningful scale for resource allocation decisions. When
    real data is loaded, this should be set from the raster metadata.
wind_speed : float
    Wind speed in km/h. Wind is the dominant driver of fire spread direction and rate (Rothermel, 1972). Modelled as a
    uniform field — per-node variation would require meteorological downscaling data not generally available at the
    resolution we operate at.
wind_direction : float
    Wind direction in degrees (0 = north, clockwise). Uniform across the landscape. Degrees picked for matching real
    data sources.
fuel_moisture : float in [0, 1]
    Relative fuel moisture. Acts as an ignition gate: high moisture suppresses spread, low moisture amplifies it. In
    practice this corresponds to the Fine Fuel Moisture Code (FFMC) from the Canadian Fire Weather Index system
    (Van Wagner, 1987).

Excluded attributes
-------------------
The following were considered and deliberately omitted:

- Canopy attributes (height, bulk density, base height, cover): relevant for crown fire modelling. We model surface fire
  only, which is appropriate for the GP resource allocation framing. Adding canopy attributes would require 3+
  additional node features with marginal benefit at this scale.
- Aspect: compass direction a slope faces. A second-order microclimate effect dominated by wind direction and slope
  magnitude at the scales we model.
- Spotting (ember transport): long-range stochastic ignition ahead of the fire front. Requires a separate sub-model and
  is out of scope for a graph-based CA.
- Dynamic fuel moisture: temporal variation in moisture during a single run. Held constant per run, consistent with
  standard probabilistic CA practice.
- Particle-level fuel properties (heat content, particle size, mineral content): Rothermel uses 13 fuel descriptors;
  categorical fuel type captures sufficient variance for our purposes.

Synthetic data
--------------
create_grid() generates a synthetic landscape from a Gaussian-smoothed terrain heightmap. Slope is derived from the
terrain gradient, giving physically consistent directionality. Low-elevation cells become water and high-slope cells
become rock (both fuel=0), matching the non-burnable categories present in real FBP and LANDFIRE data. Fuel is a
separate smoothed field, zeroed where water or rock is present.

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

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter

STATE = "state"
FUEL = "fuel"
SLOPE = "slope"
WIND_SPEED = "wind_speed"
WIND_DIRECTION = "wind_direction"
FUEL_MOISTURE = "fuel_moisture"
CELL_SIZE = "cell_size_m"


class NodeState(enum.Enum):
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    TREATED = 3


def create_grid(
    rows: int,
    cols: int,
    smoothing: float = 3.0,
    nonburnable_fraction: float = 0.0,
    cell_size_m: float = 100.0,
    seed: int | None = None,
) -> nx.Graph:
    """
    Create a synthetic landscape grid graph with spatially correlated fuel and slope.

    Slope is derived from the gradient of a synthetic terrain heightmap, giving physically consistent directionality.
    Low-elevation cells become water and high-slope cells become rock (both fuel=0). Fuel is a separate smoothed field.

    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :param smoothing: Gaussian filter sigma. Higher values produce smoother, larger-scale spatial patterns. Default 3.0
        gives patch sizes of roughly 3 nodes.
    :param nonburnable_fraction: Approximate total fraction of nodes to mark as non-burnable (fuel=0). Split evenly
        between water (lowest elevation cells) and rock (steepest cells), each receiving half the fraction. Default
        0.0 produces no non-burnable patches.
    :param cell_size_m: Side length of each grid cell in metres. Stored as a graph attribute for use by the spread model
        and real data loaders. Default 100m matches Canadian FBP operational scale.
    :param seed: Random seed for reproducibility.
    :return: Grid graph with STATE, FUEL, and SLOPE node attributes. Wind and moisture are not set.
    """
    rng = np.random.default_rng(seed)
    graph = nx.grid_2d_graph(rows, cols)

    terrain = _normalize(gaussian_filter(rng.random((rows, cols)), sigma=smoothing))
    dy, dx = np.gradient(terrain)
    slope_norm = _normalize(np.sqrt(dx**2 + dy**2))
    fuel_norm = _normalize(gaussian_filter(rng.random((rows, cols)), sigma=smoothing))

    if nonburnable_fraction > 0.0:
        half = nonburnable_fraction / 2
        fuel_norm[terrain < half] = 0.0
        fuel_norm[slope_norm > 1.0 - half] = 0.0

    graph.graph[CELL_SIZE] = cell_size_m
    _attach_node_attributes(graph, fuel_norm, slope_norm)
    return graph


def set_wind(graph: nx.Graph, speed: float, direction: float) -> None:
    """
    Set wind speed and direction as graph-level attributes.

    :param graph: The landscape graph to modify.
    :param speed: Wind speed in km/h.
    :param direction: Wind direction in degrees (0 = north, clockwise).
    """
    graph.graph[WIND_SPEED] = speed
    graph.graph[WIND_DIRECTION] = direction


def set_fuel_moisture(graph: nx.Graph, moisture: float) -> None:
    """
    Set fuel moisture as a graph-level attribute.

    :param graph: The landscape graph to modify.
    :param moisture: Relative fuel moisture in [0, 1]. 0 = bone dry, 1 = saturated.
    """
    graph.graph[FUEL_MOISTURE] = moisture


def reset_states(graph: nx.Graph) -> None:
    """
    Reset all node states to UNBURNED.

    :param graph: The landscape graph to reset.
    """
    for node in graph.nodes:
        graph.nodes[node][STATE] = NodeState.UNBURNED


def _attach_node_attributes(graph: nx.Graph, fuel_array: np.ndarray, slope_array: np.ndarray) -> None:
    for i, j in graph.nodes:
        graph.nodes[(i, j)][STATE] = NodeState.UNBURNED
        graph.nodes[(i, j)][FUEL] = float(fuel_array[i, j])
        graph.nodes[(i, j)][SLOPE] = float(slope_array[i, j])


def _normalize(array: np.ndarray) -> np.ndarray:
    low, hi = array.min(), array.max()
    if hi == low:
        return np.zeros_like(array, dtype=float)
    return (array - low) / (hi - low)
