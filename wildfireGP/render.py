"""
Landscape and simulation state rendering.

Node colours encode both the underlying terrain and the current fire state, with state taking priority over terrain for
non-UNBURNED nodes. Elevation contour lines are overlaid to give terrain context without a separate view.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from wildfireGP.network import ELEVATION, FUEL, STATE, TERRAIN, NodeState, TerrainType

_WATER = np.array([0.36, 0.61, 0.84])   # steel blue
_ROCK = np.array([0.62, 0.62, 0.62])    # medium grey
_BURNING = np.array([1.0, 0.40, 0.0])   # orange
_BURNED = np.array([0.16, 0.16, 0.16])  # charcoal
_TREATED = np.array([0.61, 0.35, 0.71]) # purple
_FUEL_CMAP = plt.cm.YlGn                # yellow-green -> dark-green by fuel load


def draw(graph: nx.Graph, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Render the landscape graph as a static image with elevation contour lines overlaid.

    State takes priority for non-UNBURNED nodes: BURNING is orange, BURNED is charcoal, TREATED is purple. For UNBURNED
    nodes, terrain type determines colour: WATER is blue, ROCK is grey, LAND uses a yellow-green to dark-green gradient
    by fuel load. White contour lines at eight elevation levels provide terrain context.

    :param graph: The landscape graph to render.
    :param ax: Axes to draw on. Creates a new figure if None.
    :return: The axes containing the rendered image.
    """
    rows, cols = _grid_dims(graph)
    elevation = _build_elevation(graph, rows, cols)

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(_build_rgb(graph, rows, cols), origin="upper")
    ax.contour(elevation, levels=8, colors="white", alpha=0.6, linewidths=0.7, origin="upper")
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
