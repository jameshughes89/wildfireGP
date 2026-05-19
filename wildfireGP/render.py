"""
Landscape and simulation state rendering.

Node colours encode both the underlying terrain and the current fire state, with state taking priority over terrain for
non-UNBURNED nodes. Elevation contour lines are overlaid to give terrain context without a separate view.
"""

import math
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

from wildfireGP.features import (
    precompute_burnable_fire_map,
    precompute_fire_map,
    precompute_reachable_unburned_area,
    precompute_state_counts,
)
from wildfireGP.network import (
    COLS,
    ELEVATION,
    FUEL,
    FUEL_MOISTURE,
    ROWS,
    STATE,
    TERRAIN,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
    TerrainType,
)

# Wind arrows point in the direction the wind is blowing (toward, not from), in 45° steps starting at north (0°).
_WIND_ARROWS = ["↓", "↙", "←", "↖", "↑", "↗", "→", "↘"]

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
    _draw_env_overlay(graph, ax)
    return ax


def animate(graphs: list[nx.Graph], path: str, fps: int = 4) -> None:
    """
    Save an animation from a sequence of landscape graph snapshots.

    Each graph in the sequence is rendered as one frame using draw(). The caller is responsible for running the
    simulation and collecting snapshots (e.g. via copy.deepcopy after each spread_step call).

    :param graphs: Ordered list of graph snapshots, one per frame.
    :param path: Output file path. Must end in .gif.
    :param fps: Frames per second. Default 4.
    """
    if not path.endswith(".gif"):
        raise ValueError(f"Only GIF output is supported; got: {path}")

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame: int) -> None:
        ax.clear()
        draw(graphs[frame], ax=ax)
        ax.set_title(f"Step {frame}")

    anim = FuncAnimation(fig, update, frames=len(graphs), interval=1000 // fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)


_SCORE_CMAP = plt.cm.RdYlGn  # red = low priority, green = high priority


def render_heatmap(
    graph: nx.Graph,
    func: Callable[[nx.Graph, tuple], float],
    path: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Render per-node priority scores of a strategy as a spatial heatmap.

    UNBURNED LAND nodes are coloured on a red-to-green scale by their score under func. Non-burnable (WATER, ROCK) and
    non-UNBURNED (BURNING, BURNED, TREATED) nodes are rendered with their standard colours matching draw(). A colorbar
    shows the score range. Elevation contour lines are overlaid.

    All four precomputes (precompute_fire_map, precompute_burnable_fire_map, precompute_reachable_unburned_area,
    precompute_state_counts) are called internally so any strategy that depends on graph-level precomputed
    features work correctly without the caller needing to do setup.

    :param graph: Landscape graph with wind and moisture set.
    :param func: Strategy callable (graph, node) -> float. Non-finite return values are treated as lowest priority.
    :param path: If given, save the figure to this path and close it.
    :param title: Optional axes title.
    :return: The axes containing the heatmap.
    """
    precompute_fire_map(graph)
    precompute_burnable_fire_map(graph)
    precompute_reachable_unburned_area(graph)
    precompute_state_counts(graph)

    rows, cols = _grid_dims(graph)
    elevation = _build_elevation(graph, rows, cols)
    rgb = _build_rgb(graph, rows, cols).copy()

    score_grid = np.full((rows, cols), np.nan)
    for i, j in graph.nodes:
        node = graph.nodes[(i, j)]
        if node[STATE] == NodeState.UNBURNED and node[TERRAIN] == TerrainType.LAND:
            s = func(graph, (i, j))
            score_grid[i, j] = s if math.isfinite(s) else np.nan

    finite_scores = score_grid[np.isfinite(score_grid)]
    if len(finite_scores) > 0:
        vmin, vmax = float(finite_scores.min()), float(finite_scores.max())
        spread = vmax - vmin if vmax != vmin else 1.0
        for i, j in graph.nodes:
            node = graph.nodes[(i, j)]
            if node[STATE] == NodeState.UNBURNED and node[TERRAIN] == TerrainType.LAND:
                v = score_grid[i, j]
                norm_v = (v - vmin) / spread if np.isfinite(v) else 0.0
                rgb[i, j] = np.array(_SCORE_CMAP(norm_v)[:3])

    _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.contour(elevation, levels=8, colors="white", alpha=0.5, linewidths=1.0)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    _draw_env_overlay(graph, ax)

    if len(finite_scores) > 0:
        sm = plt.cm.ScalarMappable(cmap=_SCORE_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="priority score")

    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    return ax


def animate_heatmap(
    graphs: list[nx.Graph],
    func: Callable[[nx.Graph, tuple], float],
    path: str,
    fps: int = 4,
    title: str | None = None,
) -> None:
    """
    Save an animated heatmap from a sequence of landscape graph snapshots.

    Each frame renders the per-node priority scores of func using render_heatmap()'s colour scheme. The score colormap
    is normalised globally across all frames so colours are comparable between timesteps --- a node that is high
    priority in frame 1 uses the same green as a high-priority node in frame 20.

    All four precomputes (precompute_fire_map, precompute_burnable_fire_map, precompute_reachable_unburned_area,
    precompute_state_counts) are called once per snapshot during a pre-scoring pass before animation begins, so
    any strategy that depends on graph-level precomputed features works correctly.

    :param graphs: Ordered list of graph snapshots, one per frame.
    :param func: Strategy callable (graph, node) -> float.
    :param path: Output file path. Must end in .gif.
    :param fps: Frames per second. Default 4.
    :param title: Optional base title; each frame appends "--- Step N".
    """
    if not path.endswith(".gif"):
        raise ValueError(f"Only GIF output is supported; got: {path}")

    for g in graphs:
        precompute_fire_map(g)
        precompute_burnable_fire_map(g)
        precompute_reachable_unburned_area(g)
        precompute_state_counts(g)

    all_finite: list[float] = []
    for g in graphs:
        for node in g.nodes:
            node_data = g.nodes[node]
            if node_data[STATE] == NodeState.UNBURNED and node_data[TERRAIN] == TerrainType.LAND:
                s = func(g, node)
                if math.isfinite(s):
                    all_finite.append(s)

    vmin = float(min(all_finite)) if all_finite else 0.0
    vmax = float(max(all_finite)) if all_finite else 1.0
    score_spread = vmax - vmin if vmax != vmin else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    sm = plt.cm.ScalarMappable(cmap=_SCORE_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="priority score")

    def update(frame: int) -> None:
        ax.clear()
        g = graphs[frame]
        rows, cols = _grid_dims(g)
        rgb = _build_rgb(g, rows, cols).copy()
        for i, j in g.nodes:
            node_data = g.nodes[(i, j)]
            if node_data[STATE] == NodeState.UNBURNED and node_data[TERRAIN] == TerrainType.LAND:
                s = func(g, (i, j))
                v = np.clip((s - vmin) / score_spread, 0.0, 1.0) if math.isfinite(s) else 0.0
                rgb[i, j] = np.array(_SCORE_CMAP(v)[:3])
        ax.imshow(rgb)
        ax.contour(_build_elevation(g, rows, cols), levels=8, colors="white", alpha=0.5, linewidths=1.0)
        ax.set_axis_off()
        frame_title = f"Step {frame}" if title is None else f"{title} --- Step {frame}"
        ax.set_title(frame_title)
        _draw_env_overlay(g, ax)

    anim = FuncAnimation(fig, update, frames=len(graphs), interval=1000 // fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)


def _draw_env_overlay(graph: nx.Graph, ax: plt.Axes) -> None:
    parts = []
    speed = graph.graph.get(WIND_SPEED)
    direction = graph.graph.get(WIND_DIRECTION)
    moisture = graph.graph.get(FUEL_MOISTURE)
    if speed is not None and direction is not None:
        arrow = _WIND_ARROWS[round(direction / 45) % 8]
        parts.append(f"{arrow} {speed:.0f} km/h")
    if moisture is not None:
        parts.append(f"moisture: {moisture:.0%}")
    if parts:
        ax.text(
            0.02,
            0.02,
            "  ".join(parts),
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5, edgecolor="none"),
        )


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
    import copy

    from wildfireGP.network import create_grid, set_fuel_moisture, set_wind
    from wildfireGP.spread import BURN_TIMER, MAX_BURN_STEPS, spread_step

    graph = create_grid(50, 50, terrain_smoothing=10, fuel_smoothing=3, water_fraction=0.05, rock_fraction=0.05)
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

    rng = np.random.default_rng()
    snapshots = [copy.deepcopy(graph)]
    for _ in range(50):
        if not any(graph.nodes[n][STATE] == NodeState.BURNING for n in graph.nodes):
            break
        spread_step(graph, rng)
        snapshots.append(copy.deepcopy(graph))

    animate(snapshots, path="spread_animation.gif", fps=4)
    print("saved spread_animation.gif")
