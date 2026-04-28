"""
Wildfire landscape graph construction and management.

This module defines the graph contract used throughout wildfireGP. A landscape is represented as a NetworkX grid
graph where each node corresponds to a spatial patch and carries three attributes: state, fuel, and slope. Fire
weather (wind and moisture) is stored as graph-level attributes, as these quantities are meteorologically driven
and effectively uniform across the landscape at the scales we model.

Node attributes
---------------
state : NodeState
    Dynamic fire state of the patch (UNBURNED, BURNING, BURNED, or TREATED). Initialised to UNBURNED; modified
    by the spread simulation.
fuel : float in [0, 1]
    Relative fuel load. Higher values indicate more burnable material and increase ignition probability and burn
    duration. On real landscapes this is derived from categorical fuel type classifications:
      - Canadian FBP system: 16 fuel types (Forestry Canada Fire Danger Group, 1992)
      - US LANDFIRE: Scott-Burgan 40 fuel models (Scott & Burgan, 2005)
    The default mappings (FBP_FUEL_MAP, SCOTT_BURGAN_FUEL_MAP) use approximate values based on known relative
    fire behavior of each fuel type. For publication-quality results, replace these with surface fuel consumption
    values (kg/m²) from the FBP tables (Forestry Canada, 1992) evaluated at a standard reference condition
    (e.g., BUI=50, ISI=10), normalized to [0, 1]. Both mappings are overrideable via the fuel_map parameter in
    the raster loading functions.
slope : float in [0, 1]
    Terrain steepness normalised to [0, 1]. Fire spread rate increases with slope (Rothermel, 1972). Derived
    from a DEM via numpy.gradient.

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

Real data loading
-----------------
from_fbp_raster() loads Canadian landscape data:
    Fuel: FBP Fuel Types GeoTIFF from NRCan/CWFIS — https://cwfis.cfs.nrcan.gc.ca/datamart
    DEM:  Canadian Digital Elevation Model (NRCan) or SRTM
    Default integer codes follow the NRCan FBP Fuel Types product (v20191114). Verify against your specific
    data product metadata and override fuel_map if codes differ.

from_landfire_raster() loads US landscape data:
    Fuel: LANDFIRE Scott-Burgan 40 fuel model GeoTIFF — https://landfire.gov/data
    DEM:  LANDFIRE elevation layer or SRTM

Both functions require rasterio. No GIS software is needed. The graph is constructed as nx.grid_2d_graph with
node (i, j) corresponding to raster pixel (row i, col j).

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
Scott, J.H. & Burgan, R.E. (2005). Standard fire behavior fuel models: a comprehensive set for use with
    Rothermel's surface fire spread model. USDA Forest Service General Technical Report RMRS-GTR-153.
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
# Default fuel type mappings
# ---------------------------------------------------------------------------

# Canadian FBP fuel types -> relative fuel load [0, 1].
# Integer codes from NRCan FBP Fuel Types GeoTIFF (v20191114).
# Verify against your specific data product and override if codes differ.
#
# Values are approximate, based on known relative fire behavior of each type. For publication-quality work,
# replace with surface fuel consumption (kg/m²) from the FBP system tables (Forestry Canada Fire Danger
# Group, 1992) at a standard reference condition (e.g., BUI=50, ISI=10), normalized to [0, 1].
FBP_FUEL_MAP: dict[int, float] = {
    0: 0.0,   # Non-fuel (water, rock, urban, agriculture)
    1: 0.5,   # C-1: Spruce-Lichen Woodland
    2: 0.8,   # C-2: Boreal Spruce
    3: 1.0,   # C-3: Mature Jack or Lodgepole Pine
    4: 1.0,   # C-4: Immature Jack or Lodgepole Pine
    5: 0.7,   # C-5: Red and White Pine
    6: 0.9,   # C-6: Conifer Plantation
    7: 0.6,   # C-7: Ponderosa Pine - Douglas-Fir
    8: 0.4,   # D-1: Leafless Aspen
    9: 0.5,   # D-2: Green Aspen
    11: 0.7,  # M-1: Boreal Mixedwood - Leafless
    12: 0.7,  # M-2: Boreal Mixedwood - Green
    13: 0.8,  # M-3: Dead Balsam Fir - Mixedwood
    14: 0.8,  # M-4: Dead Balsam Fir - Mixedwood (Green)
    15: 0.6,  # O-1a: Matted Grass
    16: 0.6,  # O-1b: Standing Grass
    17: 0.5,  # S-1: Jack or Lodgepole Pine Slash
    18: 0.6,  # S-2: White Spruce - Balsam Slash
    19: 0.7,  # S-3: Coastal Cedar - Hemlock - Douglas-Fir Slash
}

# US LANDFIRE Scott-Burgan 40 fuel models -> relative fuel load [0, 1].
# Integer codes are the standard LANDFIRE fuel model numbers.
#
# Values are approximate. For publication-quality work, replace with fuel load values from Scott & Burgan
# (2005) Table 1 (total fuel load, kg/m²), normalized to [0, 1].
SCOTT_BURGAN_FUEL_MAP: dict[int, float] = {
    # Non-burnable
    91: 0.0, 92: 0.0, 93: 0.0, 98: 0.0, 99: 0.0,
    # Grass (GR)
    101: 0.3, 102: 0.4, 103: 0.5, 104: 0.5,
    105: 0.6, 106: 0.6, 107: 0.7, 108: 0.7, 109: 0.7,
    # Grass-shrub (GS)
    121: 0.5, 122: 0.6, 123: 0.7, 124: 0.7,
    # Shrub (SH)
    141: 0.5, 142: 0.6, 143: 0.7, 144: 0.7,
    145: 0.7, 146: 0.8, 147: 0.8, 148: 0.8, 149: 0.8,
    # Timber understory (TU)
    161: 0.5, 162: 0.6, 163: 0.7, 164: 0.7, 165: 0.7,
    # Timber litter (TL)
    181: 0.6, 182: 0.7, 183: 0.7, 184: 0.8,
    185: 0.8, 186: 0.8, 187: 0.9, 188: 0.9, 189: 0.9,
    # Slash-blowdown (SB)
    201: 0.7, 202: 0.8, 203: 0.9, 204: 0.9,
}


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
# Real data loading
# ---------------------------------------------------------------------------


def from_fbp_raster(
    fuel_path: str,
    dem_path: str,
    fuel_map: dict[int, float] | None = None,
) -> nx.Graph:
    """Build a landscape graph from Canadian FBP fuel type and DEM rasters.

    Parameters
    ----------
    fuel_path : str
        Path to NRCan/CWFIS FBP Fuel Types GeoTIFF.
    dem_path : str
        Path to DEM GeoTIFF (NRCan CDEM or SRTM).
    fuel_map : dict or None
        Mapping from FBP integer code to float [0, 1]. Defaults to FBP_FUEL_MAP. Unknown codes map to 0.0.

    Returns
    -------
    nx.Graph
        Grid graph with STATE, FUEL, and SLOPE node attributes.
    """
    import rasterio

    mapping = fuel_map if fuel_map is not None else FBP_FUEL_MAP

    with rasterio.open(fuel_path) as src:
        raw = src.read(1).astype(int)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)

    fuel_array = np.vectorize(lambda x: mapping.get(x, 0.0))(raw)
    return _build_graph_from_arrays(fuel_array, dem)


def from_landfire_raster(
    fuel_path: str,
    dem_path: str,
    fuel_map: dict[int, float] | None = None,
) -> nx.Graph:
    """Build a landscape graph from US LANDFIRE Scott-Burgan fuel and DEM rasters.

    Parameters
    ----------
    fuel_path : str
        Path to LANDFIRE Scott-Burgan 40 fuel model GeoTIFF.
    dem_path : str
        Path to DEM GeoTIFF (LANDFIRE elevation or SRTM).
    fuel_map : dict or None
        Mapping from Scott-Burgan integer code to float [0, 1]. Defaults to SCOTT_BURGAN_FUEL_MAP.
        Unknown codes map to 0.0.

    Returns
    -------
    nx.Graph
        Grid graph with STATE, FUEL, and SLOPE node attributes.
    """
    import rasterio

    mapping = fuel_map if fuel_map is not None else SCOTT_BURGAN_FUEL_MAP

    with rasterio.open(fuel_path) as src:
        raw = src.read(1).astype(int)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)

    fuel_array = np.vectorize(lambda x: mapping.get(x, 0.0))(raw)
    return _build_graph_from_arrays(fuel_array, dem)


def _build_graph_from_arrays(fuel_array: np.ndarray, dem_array: np.ndarray) -> nx.Graph:
    rows, cols = fuel_array.shape
    graph = nx.grid_2d_graph(rows, cols)

    dy, dx = np.gradient(dem_array)
    slope_norm = _normalize(np.sqrt(dx**2 + dy**2))

    _attach_node_attributes(graph, fuel_array, slope_norm)
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
