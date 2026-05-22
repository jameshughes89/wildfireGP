"""
Landscape and simulation state rendering.

Node colours encode both the underlying terrain and the current fire state, with state taking priority over terrain
for non-UNBURNED nodes. Elevation contour lines are overlaid to give terrain context without a separate view.
"""

import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from wildfireGP.features import (
    precompute_burnable_fire_map,
    precompute_fire_map,
    precompute_reachable_unburned_area,
    precompute_state_counts,
)
from wildfireGP.network import GraphState, NodeState, TerrainType

_WIND_ARROWS = ["↓", "↙", "←", "↖", "↑", "↗", "→", "↘"]

_WATER = np.array([0.36, 0.61, 0.84])
_ROCK = np.array([0.62, 0.62, 0.62])
_BURNING = np.array([1.0, 0.40, 0.0])
_BURNED = np.array([0.16, 0.16, 0.16])
_TREATED = np.array([0.61, 0.35, 0.71])
_FUEL_CMAP = plt.cm.YlGn


def draw(state: GraphState, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Render the landscape state as a static image with elevation contour lines overlaid.
    """
    elevation = state.elevation

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(_build_rgb(state))
    ax.contour(elevation, levels=8, colors="white", alpha=0.5, linewidths=1.0)
    ax.set_axis_off()
    _draw_env_overlay(state, ax)
    return ax


def animate(states: list[GraphState], path: str, fps: int = 4) -> None:
    """
    Save an animation from a sequence of landscape state snapshots.
    """
    if not path.endswith(".gif"):
        raise ValueError(f"Only GIF output is supported; got: {path}")

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame: int) -> None:
        ax.clear()
        draw(states[frame], ax=ax)
        ax.set_title(f"Step {frame}")

    anim = FuncAnimation(fig, update, frames=len(states), interval=1000 // fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)


_SCORE_CMAP = plt.cm.RdYlGn


def render_heatmap(
    state: GraphState,
    func: Callable[[GraphState, tuple], float],
    path: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Render per-node priority scores of a strategy as a spatial heatmap.

    All four precomputes are called internally so any strategy that depends on graph-level precomputed features works
    correctly without the caller needing to do setup.
    """
    precompute_fire_map(state)
    precompute_burnable_fire_map(state)
    precompute_reachable_unburned_area(state)
    precompute_state_counts(state)

    rows, cols = state.rows, state.cols
    rgb = _build_rgb(state).copy()

    score_grid = np.full((rows, cols), np.nan)
    burnable = (state.state == NodeState.UNBURNED) & (state.terrain == TerrainType.LAND)
    for i, j in np.argwhere(burnable):
        s = func(state, (int(i), int(j)))
        score_grid[i, j] = s if math.isfinite(s) else np.nan

    finite_scores = score_grid[np.isfinite(score_grid)]
    if len(finite_scores) > 0:
        vmin, vmax = float(finite_scores.min()), float(finite_scores.max())
        spread = vmax - vmin if vmax != vmin else 1.0
        for i, j in np.argwhere(burnable):
            v = score_grid[i, j]
            norm_v = (v - vmin) / spread if np.isfinite(v) else 0.0
            rgb[i, j] = np.array(_SCORE_CMAP(norm_v)[:3])

    _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.contour(state.elevation, levels=8, colors="white", alpha=0.5, linewidths=1.0)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    _draw_env_overlay(state, ax)

    if len(finite_scores) > 0:
        sm = plt.cm.ScalarMappable(cmap=_SCORE_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="priority score")

    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    return ax


def animate_heatmap(
    states: list[GraphState],
    func: Callable[[GraphState, tuple], float],
    path: str,
    fps: int = 4,
    title: str | None = None,
) -> None:
    """
    Save an animated heatmap from a sequence of landscape state snapshots.

    The score colormap is normalised globally across all frames so colours are comparable between timesteps.
    """
    if not path.endswith(".gif"):
        raise ValueError(f"Only GIF output is supported; got: {path}")

    for s in states:
        precompute_fire_map(s)
        precompute_burnable_fire_map(s)
        precompute_reachable_unburned_area(s)
        precompute_state_counts(s)

    all_finite: list[float] = []
    for s in states:
        burnable = (s.state == NodeState.UNBURNED) & (s.terrain == TerrainType.LAND)
        for i, j in np.argwhere(burnable):
            score = func(s, (int(i), int(j)))
            if math.isfinite(score):
                all_finite.append(score)

    vmin = float(min(all_finite)) if all_finite else 0.0
    vmax = float(max(all_finite)) if all_finite else 1.0
    score_spread = vmax - vmin if vmax != vmin else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    sm = plt.cm.ScalarMappable(cmap=_SCORE_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="priority score")

    def update(frame: int) -> None:
        ax.clear()
        s = states[frame]
        rgb = _build_rgb(s).copy()
        burnable = (s.state == NodeState.UNBURNED) & (s.terrain == TerrainType.LAND)
        for i, j in np.argwhere(burnable):
            score = func(s, (int(i), int(j)))
            v = np.clip((score - vmin) / score_spread, 0.0, 1.0) if math.isfinite(score) else 0.0
            rgb[i, j] = np.array(_SCORE_CMAP(v)[:3])
        ax.imshow(rgb)
        ax.contour(s.elevation, levels=8, colors="white", alpha=0.5, linewidths=1.0)
        ax.set_axis_off()
        frame_title = f"Step {frame}" if title is None else f"{title} --- Step {frame}"
        ax.set_title(frame_title)
        _draw_env_overlay(s, ax)

    anim = FuncAnimation(fig, update, frames=len(states), interval=1000 // fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)


def _draw_env_overlay(state: GraphState, ax: plt.Axes) -> None:
    parts = []
    speed = state.wind_speed
    direction = state.wind_direction
    moisture = state.fuel_moisture
    arrow = _WIND_ARROWS[round(direction / 45) % 8]
    parts.append(f"{arrow} {speed:.0f} km/h")
    parts.append(f"moisture: {moisture:.0%}")
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


def _build_rgb(state: GraphState) -> np.ndarray:
    rows, cols = state.rows, state.cols
    rgb = np.zeros((rows, cols, 3))
    state_arr = state.state
    terrain_arr = state.terrain
    fuel_arr = state.fuel
    for i in range(rows):
        for j in range(cols):
            s = state_arr[i, j]
            t = terrain_arr[i, j]
            if s == NodeState.BURNING:
                color = _BURNING
            elif s == NodeState.BURNED:
                color = _BURNED
            elif s == NodeState.TREATED:
                color = _TREATED
            elif t == TerrainType.WATER:
                color = _WATER
            elif t == TerrainType.ROCK:
                color = _ROCK
            else:
                color = np.array(_FUEL_CMAP(float(fuel_arr[i, j]))[:3])
            rgb[i, j] = color
    return rgb


if __name__ == "__main__":
    from wildfireGP.network import create_grid, set_fuel_moisture, set_wind
    from wildfireGP.spread import MAX_BURN_STEPS, spread_step

    state = create_grid(50, 50, terrain_smoothing=10, fuel_smoothing=3, water_fraction=0.05, rock_fraction=0.05)
    set_wind(state, speed=20.0, direction=45.0)
    set_fuel_moisture(state, moisture=0.1)

    ignition_node = (25, 25)
    state.state[ignition_node] = NodeState.BURNING
    state.burn_timer[ignition_node] = max(1, math.ceil(float(state.fuel[ignition_node]) * MAX_BURN_STEPS))

    fig, ax_map = plt.subplots(figsize=(8, 8))
    draw(state, ax=ax_map)
    plt.tight_layout()
    plt.savefig("draw_test.png", dpi=150)
    print("saved draw_test.png")

    rng = np.random.default_rng()
    snapshots = [state.copy()]
    for _ in range(50):
        if not (state.state == NodeState.BURNING).any():
            break
        spread_step(state, rng)
        snapshots.append(state.copy())

    animate(snapshots, path="spread_animation.gif", fps=4)
    print("saved spread_animation.gif")
