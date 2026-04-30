"""
Landscape and simulation state rendering.

Node colours encode both the underlying terrain and the current fire state, with state taking priority over terrain for
non-UNBURNED nodes. Elevation contour lines are overlaid to give terrain context without a separate view.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

from wildfireGP.network import (
    COLS,
    ELEVATION,
    FUEL,
    ROWS,
    STATE,
    TERRAIN,
    NodeState,
    TerrainType,
)

_WATER = np.array([0.36, 0.61, 0.84])  # steel blue
_ROCK = np.array([0.62, 0.62, 0.62])  # medium grey
_BURNING = np.array([1.0, 0.40, 0.0])  # orange
_BURNED = np.array([0.16, 0.16, 0.16])  # charcoal
_TREATED = np.array([0.61, 0.35, 0.71])  # purple
_FUEL_CMAP = plt.cm.YlGn  # yellow-green -> dark-green by fuel load


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

    ax.imshow(_build_rgb(graph, rows, cols))
    ax.contour(elevation, levels=8, colors="white", alpha=0.5, linewidths=1.0)
    ax.set_axis_off()
    return ax


def animate(graph: nx.Graph, steps: int, rng: np.random.Generator, path: str, fps: int = 4) -> None:
    """
    Run the fire spread simulation and save an animation.

    Advances the simulation one timestep per frame, stopping early if fire goes out. The graph is modified in place.

    :param graph: Landscape graph with WIND_SPEED, WIND_DIRECTION, and FUEL_MOISTURE set, and at least one BURNING node.
    :param steps: Maximum number of timesteps (frames) to render.
    :param rng: NumPy random generator for the spread model.
    :param path: Output file path. Extension determines format (.gif, .mp4).
    :param fps: Frames per second. Default 4.
    """
    from wildfireGP.spread import spread_step

    fig, ax = plt.subplots(figsize=(8, 8))
    step_counter = [0]

    def update(_frame: int) -> None:
        ax.clear()
        draw(graph, ax=ax)
        ax.set_title(f"Step {step_counter[0]}")
        step_counter[0] += 1
        if any(graph.nodes[n][STATE] == NodeState.BURNING for n in graph.nodes):
            spread_step(graph, rng)

    anim = FuncAnimation(fig, update, frames=steps, interval=1000 // fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)


def _grid_dims(graph: nx.Graph) -> tuple[int, int]:
    return graph.graph[ROWS], graph.graph[COLS]


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


if __name__ == "__main__":
    import math

    from wildfireGP.network import NodeState, create_grid, set_fuel_moisture, set_wind
    from wildfireGP.spread import MAX_BURN_STEPS, BURN_TIMER

    graph = create_grid(50, 50, terrain_smoothing=10, fuel_smoothing=3, water_fraction=0.05, rock_fraction=0.05, seed=42)
    set_wind(graph, speed=20.0, direction=45.0)
    set_fuel_moisture(graph, moisture=0.1)

    ignition_node = (25, 25)
    graph.nodes[ignition_node][STATE] = NodeState.BURNING
    graph.nodes[ignition_node][BURN_TIMER] = max(1, math.ceil(graph.nodes[ignition_node][FUEL] * MAX_BURN_STEPS))

    fig, ax_map = plt.subplots(figsize=(8, 8))
    draw(graph, ax=ax_map)
    plt.tight_layout()
    plt.savefig("draw_test.png", dpi=150)
    print("saved draw_test.png")

    graph2 = create_grid(50, 50, terrain_smoothing=10, fuel_smoothing=3, water_fraction=0.05, rock_fraction=0.05, seed=42)
    set_wind(graph2, speed=20.0, direction=45.0)
    set_fuel_moisture(graph2, moisture=0.1)
    graph2.nodes[ignition_node][STATE] = NodeState.BURNING
    graph2.nodes[ignition_node][BURN_TIMER] = max(1, math.ceil(graph2.nodes[ignition_node][FUEL] * MAX_BURN_STEPS))

    rng = np.random.default_rng(42)
    animate(graph2, steps=40, rng=rng, path="spread_animation.gif", fps=4)
    print("saved spread_animation.gif")
