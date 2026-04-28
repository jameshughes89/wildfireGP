"""
Landscape and simulation state rendering.

Node colours encode both the underlying terrain and the current fire state, with state taking priority over terrain for
non-UNBURNED nodes.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from wildfireGP.network import FUEL, SLOPE, STATE, NodeState

_WATER = np.array([0.36, 0.61, 0.84])   # steel blue
_ROCK = np.array([0.62, 0.62, 0.62])    # medium grey
_BURNING = np.array([1.0, 0.40, 0.0])   # orange
_BURNED = np.array([0.16, 0.16, 0.16])  # charcoal
_TREATED = np.array([0.61, 0.35, 0.71]) # purple
_FUEL_CMAP = plt.cm.YlGn                # yellow-green -> dark-green by fuel load


def draw(graph: nx.Graph, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Render the landscape graph as a static image.

    State takes priority for non-UNBURNED nodes: BURNING is orange, BURNED is charcoal, TREATED is purple. For UNBURNED 
    nodes, fuel=0 cells are rendered as water (blue) or rock (grey) based on slope - high slope implies rock - and 
    burnable cells use a yellow-green to dark-green gradient by fuel load. WARNING: The cutoff can cause issues with
    water/rock rendering if water values in network are high.

    :param graph: The landscape graph to render.
    :param ax: Axes to draw on. Creates a new figure if None.
    :return: The axes containing the rendered image.
    """
    rows = max(i for i, j in graph.nodes) + 1
    cols = max(j for i, j in graph.nodes) + 1

    rgb = np.zeros((rows, cols, 3))

    for i, j in graph.nodes:
        node = graph.nodes[(i, j)]
        state = node[STATE]
        fuel = node[FUEL]
        slope = node[SLOPE]

        if state == NodeState.BURNING:
            color = _BURNING
        elif state == NodeState.BURNED:
            color = _BURNED
        elif state == NodeState.TREATED:
            color = _TREATED
        elif fuel == 0.0:
            color = _ROCK if slope > 0.5 else _WATER
        else:
            color = np.array(_FUEL_CMAP(fuel)[:3])

        rgb[i, j] = color

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()
    return ax
