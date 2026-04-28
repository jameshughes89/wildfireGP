"""
Landscape and simulation state rendering.

Node colours encode both the underlying terrain and the current fire state, with state taking priority over terrain for
non-UNBURNED nodes. Four render functions are provided, each visualising a different aspect of the landscape.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LightSource

from wildfireGP.network import ELEVATION, FUEL, STATE, TERRAIN, NodeState, TerrainType

_WATER = np.array([0.36, 0.61, 0.84])   # steel blue
_ROCK = np.array([0.62, 0.62, 0.62])    # medium grey
_BURNING = np.array([1.0, 0.40, 0.0])   # orange
_BURNED = np.array([0.16, 0.16, 0.16])  # charcoal
_TREATED = np.array([0.61, 0.35, 0.71]) # purple
_FUEL_CMAP = plt.cm.YlGn                # yellow-green -> dark-green by fuel load
_ELEV_CMAP = plt.cm.terrain             # elevation colormap for draw_elevation


def draw(graph: nx.Graph, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Render the landscape graph as a static image.

    State takes priority for non-UNBURNED nodes: BURNING is orange, BURNED is charcoal, TREATED is purple. For UNBURNED
    nodes, terrain type determines colour: WATER is blue, ROCK is grey, LAND uses a yellow-green to dark-green gradient
    by fuel load.

    :param graph: The landscape graph to render.
    :param ax: Axes to draw on. Creates a new figure if None.
    :return: The axes containing the rendered image.
    """
    rows, cols = _grid_dims(graph)

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(_build_rgb(graph, rows, cols), origin="upper")
    ax.set_axis_off()
    return ax


def draw_hillshaded(graph: nx.Graph, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Render the landscape with hillshading blended into the fuel and terrain colours.

    A simulated northwest light source shades the terrain based on elevation, making ridges and valleys immediately
    readable. The colour scheme is identical to draw(); hillshading only modulates brightness.

    :param graph: The landscape graph to render.
    :param ax: Axes to draw on. Creates a new figure if None.
    :return: The axes containing the rendered image.
    """
    rows, cols = _grid_dims(graph)
    rgb = _build_rgb(graph, rows, cols)
    elevation = _build_elevation(graph, rows, cols)

    intensity = LightSource(azdeg=315, altdeg=45).hillshade(elevation, vert_exag=2.0)
    intensity = 0.3 + 0.7 * intensity  # scale to [0.3, 1.0] so shadows are never fully black
    rgb = np.clip(rgb * intensity[:, :, np.newaxis], 0, 1)

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()
    return ax


def draw_contours(graph: nx.Graph, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Render the landscape with elevation contour lines overlaid.

    The base render is identical to draw(). White contour lines are drawn on top at eight elevation levels,
    giving a topographic map feel.

    :param graph: The landscape graph to render.
    :param ax: Axes to draw on. Creates a new figure if None.
    :return: The axes containing the rendered image.
    """
    rows, cols = _grid_dims(graph)
    elevation = _build_elevation(graph, rows, cols)

    ax = draw(graph, ax=ax)
    ax.contour(elevation, levels=8, colors="white", alpha=0.5, linewidths=0.5, origin="upper")
    return ax


def draw_elevation(graph: nx.Graph, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Render elevation as a standalone bubble map.

    Each node is drawn as a square marker sized and coloured by elevation. Higher nodes are larger and brighter.
    Fire state and fuel are not shown; this view is purely for inspecting terrain structure.

    :param graph: The landscape graph to render.
    :param ax: Axes to draw on. Creates a new figure if None.
    :return: The axes containing the rendered image.
    """
    rows, cols = _grid_dims(graph)

    is_, js, elevations = [], [], []
    for i, j in graph.nodes:
        is_.append(i)
        js.append(j)
        elevations.append(graph.nodes[(i, j)][ELEVATION])

    elevations = np.array(elevations)
    sizes = 5 + elevations * 50  # scale marker area from 5 to 55

    if ax is None:
        _, ax = plt.subplots()

    ax.set_facecolor("#1a1a1a")
    ax.scatter(js, is_, s=sizes, c=elevations, cmap=_ELEV_CMAP, marker="s", linewidths=0)
    ax.set_xlim(-1, cols)
    ax.set_ylim(rows, -1)
    ax.set_axis_off()
    return ax


def _grid_dims(graph: nx.Graph) -> tuple[int, int]:
    rows = max(i for i, j in graph.nodes) + 1
    cols = max(j for i, j in graph.nodes) + 1
    return rows, cols


def _build_rgb(graph: nx.Graph, rows: int, cols: int) -> np.ndarray:
    rgb = np.zeros((rows, cols, 3))
    for i, j in graph.nodes:
        node = graph.nodes[(i, j)]
        state = node[STATE]
        terrain = node[TERRAIN]
        fuel = node[FUEL]

        if state == NodeState.BURNING:
            color = _BURNING
        elif state == NodeState.BURNED:
            color = _BURNED
        elif state == NodeState.TREATED:
            color = _TREATED
        elif terrain == TerrainType.WATER:
            color = _WATER
        elif terrain == TerrainType.ROCK:
            color = _ROCK
        else:
            color = np.array(_FUEL_CMAP(fuel)[:3])

        rgb[i, j] = color
    return rgb


def _build_elevation(graph: nx.Graph, rows: int, cols: int) -> np.ndarray:
    elevation = np.zeros((rows, cols))
    for i, j in graph.nodes:
        elevation[i, j] = graph.nodes[(i, j)][ELEVATION]
    return elevation
