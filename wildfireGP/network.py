"""
Wildfire landscape graph construction and management.

A landscape is represented as a NetworkX grid graph where each node corresponds to a spatial patch and carries six
attributes: state, burn_timer, terrain, fuel, slope, and elevation. Fire weather (wind and moisture) is stored as
graph-level attributes, as these quantities are meteorologically driven and effectively uniform across the landscape at
the scales we model. Nodes use a Moore (8-connectivity) neighbourhood: each cell is connected to its four cardinal
neighbours and four diagonal neighbours, matching the Alexandridis et al. (2008) fire spread model.

Node attributes
---------------
state : NodeState
    Dynamic fire state of the patch (UNBURNED, BURNING, BURNED, or TREATED). Initialised to UNBURNED; modified by the
    spread simulation.
burn_timer : int >= 0
    Remaining timesteps the node will continue burning. Set to ceil(fuel * MAX_BURN_STEPS) on ignition; decremented each
    timestep; node transitions to BURNED when it reaches zero. Initialised to 0 for all nodes and maintained as internal
    spread-model state rather than a GP feature.
terrain : TerrainType
    Physical terrain category: LAND (burnable), WATER (low-elevation basin), or ROCK (steep slope). WATER and ROCK nodes
    have fuel=0 and are non-burnable. Stored explicitly so the spread model and GP terminals can distinguish terrain
    type without re-inferring it from elevation or slope.
fuel : float in [0, 1]
    Relative fuel load. Higher values indicate more burnable material and increase ignition probability and burn
    duration. Currently assigned via spatially correlated synthetic generation. When real data loading is added, this
    will be derived from categorical fuel type classifications (Canadian FBP system or US LANDFIRE Scott-Burgan models).
slope : float in [0, 1]
    Terrain steepness normalised to [0, 1]. Fire spread rate increases with slope (Rothermel, 1972). Derived from a
    synthetic terrain heightmap via numpy.gradient; will use a real DEM when real data loading is added.
elevation : float in [0, 1]
    Relative terrain height normalised to [0, 1]. Elevation is the primary terrain variable from which slope and
    non-burnable patches are derived: low-elevation basins become water, and steep slopes derived from the terrain
    gradient become rock. Synthetically generated as the heightmap itself; will be derived from a DEM when real data
    loading is added.

Graph-level attributes
----------------------
cell_size_m : float
    Side length of each grid cell in metres. Defaults to 100m, matching Canadian FBP operational scale. A 50x50 grid at
    100m represents a 5km x 5km landscape, which is a meaningful scale for resource allocation decisions. When real data
    is loaded, this should be set from the raster metadata.
wind_speed : float
    Wind speed in km/h. Wind is the dominant driver of fire spread direction and rate (Rothermel, 1972). Modelled as a
    uniform field --- per-node variation would require meteorological downscaling data not generally available at the
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
separate smoothed field, zeroed where water or rock is present. Terrain and fuel smoothing are independent parameters,
allowing rugged terrain with fine-grained fuel variation or smooth terrain with coarse fuel patches.

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
TERRAIN = "terrain"
FUEL = "fuel"
SLOPE = "slope"
ELEVATION = "elevation"
WIND_SPEED = "wind_speed"
WIND_DIRECTION = "wind_direction"
FUEL_MOISTURE = "fuel_moisture"
CELL_SIZE = "cell_size_m"
ROWS = "rows"
COLS = "cols"
BURN_TIMER = "burn_timer"


class NodeState(enum.Enum):
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    TREATED = 3


class TerrainType(enum.Enum):
    LAND = 0
    WATER = 1
    ROCK = 2


def create_grid(
    rows: int,
    cols: int,
    terrain_smoothing: float = 3.0,
    fuel_smoothing: float = 3.0,
    water_fraction: float = 0.0,
    rock_fraction: float = 0.0,
    cell_size_m: float = 100.0,
    seed: int | None = None,
) -> nx.Graph:
    """
    Create a synthetic landscape grid graph with spatially correlated fuel and slope.

    Slope is derived from the gradient of a synthetic terrain heightmap, giving physically consistent directionality.
    Low-elevation cells become water and high-slope cells become rock (both fuel=0). Fuel is a separate smoothed field.
    Terrain and fuel smoothing are independent: rugged terrain with coarse fuel patches and smooth terrain with
    fine-grained fuel variation are both representable.

    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :param terrain_smoothing: Gaussian filter sigma for the terrain heightmap. Controls the spatial scale of hills,
        valleys, and ridges. Higher values produce broader, more gradual terrain.
    :param fuel_smoothing: Gaussian filter sigma for the fuel field. Controls the spatial scale of fuel patches. Higher
        values produce larger, more homogeneous fuel zones.
    :param water_fraction: Fraction of nodes to mark as water (fuel=0), selected from the lowest-elevation cells.
        Default 0.0 produces no water.
    :param rock_fraction: Fraction of nodes to mark as rock (fuel=0), selected from the steepest cells. Default 0.0
        produces no rock.
    :param cell_size_m: Side length of each grid cell in metres. Stored as a graph attribute for use by the spread model
        and real data loaders. Default 100m matches Canadian FBP operational scale.
    :param seed: Random seed for reproducibility.
    :return: Grid graph with STATE, TERRAIN, FUEL, SLOPE, and ELEVATION node attributes. Wind and moisture are not set.
        Nodes are connected with a Moore (8-connectivity) neighbourhood.
    """
    rng = np.random.default_rng(seed)
    graph = nx.grid_2d_graph(rows, cols)
    for i in range(rows - 1):
        for j in range(cols - 1):
            graph.add_edge((i, j), (i + 1, j + 1))
            graph.add_edge((i + 1, j), (i, j + 1))

    terrain_height = _normalize(gaussian_filter(rng.random((rows, cols)), sigma=terrain_smoothing))
    dy, dx = np.gradient(terrain_height)
    slope_norm = _normalize(np.sqrt(dx**2 + dy**2))
    fuel_norm = _normalize(gaussian_filter(rng.random((rows, cols)), sigma=fuel_smoothing))

    terrain_type = np.full((rows, cols), TerrainType.LAND, dtype=object)
    if water_fraction > 0.0:
        mask = terrain_height < water_fraction
        fuel_norm[mask] = 0.0
        terrain_type[mask] = TerrainType.WATER
    if rock_fraction > 0.0:
        mask = slope_norm > 1.0 - rock_fraction
        fuel_norm[mask] = 0.0
        terrain_type[mask] = TerrainType.ROCK

    graph.graph[CELL_SIZE] = cell_size_m
    graph.graph[ROWS] = rows
    graph.graph[COLS] = cols
    _attach_node_attributes(graph, fuel_norm, slope_norm, terrain_height, terrain_type)
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
        graph.nodes[node][BURN_TIMER] = 0


def _attach_node_attributes(
    graph: nx.Graph,
    fuel_array: np.ndarray,
    slope_array: np.ndarray,
    elevation_array: np.ndarray,
    terrain_type_array: np.ndarray,
) -> None:
    for i, j in graph.nodes:
        graph.nodes[(i, j)][STATE] = NodeState.UNBURNED
        graph.nodes[(i, j)][BURN_TIMER] = 0
        graph.nodes[(i, j)][TERRAIN] = terrain_type_array[i, j]
        graph.nodes[(i, j)][FUEL] = float(fuel_array[i, j])
        graph.nodes[(i, j)][SLOPE] = float(slope_array[i, j])
        graph.nodes[(i, j)][ELEVATION] = float(elevation_array[i, j])


def _normalize(array: np.ndarray) -> np.ndarray:
    low, hi = array.min(), array.max()
    if hi == low:
        return np.zeros_like(array, dtype=float)
    return (array - low) / (hi - low)
