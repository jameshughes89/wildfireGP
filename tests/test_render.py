import math

import matplotlib.pyplot as plt
import numpy as np

from wildfireGP.network import (
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.render import (
    _build_rgb,
    animate,
    animate_heatmap,
    draw,
    render_heatmap,
)
from wildfireGP.spread import MAX_BURN_STEPS, spread_step

_NODE = (1, 1)


def _state_with_cell():
    return create_grid(3, 3, seed=0)


def test_draw_default_returns_axes():
    s = create_grid(5, 5, seed=0)
    ax = draw(s)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_draw_with_provided_ax_returns_same_ax():
    s = create_grid(5, 5, seed=0)
    fig, ax = plt.subplots()
    result = draw(s, ax=ax)
    plt.close(fig)
    assert result is ax


def test_build_rgb_state_takes_priority_over_terrain():
    burning_land = _state_with_cell()
    burning_land.state[_NODE] = NodeState.BURNING

    burning_water = _state_with_cell()
    burning_water.state[_NODE] = NodeState.BURNING
    burning_water.terrain[_NODE] = TerrainType.WATER

    water = _state_with_cell()
    water.terrain[_NODE] = TerrainType.WATER

    assert np.array_equal(_build_rgb(burning_water)[_NODE], _build_rgb(burning_land)[_NODE])
    assert not np.array_equal(_build_rgb(burning_water)[_NODE], _build_rgb(water)[_NODE])


def _snapshots(seed=0, steps=3):
    s = create_grid(5, 5, seed=seed)
    set_wind(s, speed=20.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.1)
    node = (2, 2)
    s.state[node] = NodeState.BURNING
    s.burn_timer[node] = max(1, math.ceil(float(s.fuel[node]) * MAX_BURN_STEPS))
    rng = np.random.default_rng(seed)
    frames = [s.copy()]
    for _ in range(steps - 1):
        spread_step(s, rng)
        frames.append(s.copy())
    return frames


def test_animate_creates_gif(tmp_path):
    out = tmp_path / "out.gif"
    animate(_snapshots(), path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_does_not_leave_open_figures(tmp_path):
    before = len(plt.get_fignums())
    animate(_snapshots(), path=str(tmp_path / "out.gif"))
    assert len(plt.get_fignums()) == before


def test_animate_single_frame(tmp_path):
    s = create_grid(3, 3, seed=0)
    out = tmp_path / "out.gif"
    animate([s], path=str(out))
    assert out.exists() and out.stat().st_size > 0


def _heatmap_state():
    s = create_grid(5, 5, seed=0)
    set_wind(s, speed=10.0, direction=0.0)
    set_fuel_moisture(s, moisture=0.2)
    return s


def test_render_heatmap_returns_axes():
    s = _heatmap_state()
    ax = render_heatmap(s, lambda state, node: 1.0)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_render_heatmap_saves_file(tmp_path):
    s = _heatmap_state()
    out = tmp_path / "heatmap.png"
    render_heatmap(s, lambda state, node: 1.0, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_heatmap_handles_all_nonfinite_scores():
    s = _heatmap_state()
    ax = render_heatmap(s, lambda state, node: float("-inf"))
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_render_heatmap_handles_uniform_scores():
    s = _heatmap_state()
    ax = render_heatmap(s, lambda state, node: 0.5)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_animate_heatmap_creates_gif(tmp_path):
    out = tmp_path / "heatmap.gif"
    animate_heatmap(_snapshots(), lambda state, node: 1.0, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_heatmap_single_frame(tmp_path):
    s = _heatmap_state()
    out = tmp_path / "out.gif"
    animate_heatmap([s], lambda state, node: 1.0, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_heatmap_handles_all_nonfinite_scores(tmp_path):
    out = tmp_path / "out.gif"
    animate_heatmap(_snapshots(), lambda state, node: float("-inf"), path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_heatmap_global_normalisation_uses_all_frames(tmp_path):
    scores = iter(range(100))

    def incrementing(state, node):
        return float(next(scores, 0))

    out = tmp_path / "out.gif"
    animate_heatmap(_snapshots(steps=3), incrementing, path=str(out))
    assert out.exists() and out.stat().st_size > 0
