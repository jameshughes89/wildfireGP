import copy
import math

import matplotlib.pyplot as plt
import numpy as np

from wildfireGP.network import (
    BURN_TIMER,
    ELEVATION,
    FUEL,
    STATE,
    TERRAIN,
    NodeState,
    TerrainType,
    create_grid,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.render import (
    _BURNED,
    _BURNING,
    _ROCK,
    _TREATED,
    _WATER,
    _build_elevation,
    _build_rgb,
    animate,
    animate_heatmap,
    draw,
    render_heatmap,
)
from wildfireGP.spread import MAX_BURN_STEPS, spread_step

_NODE = (1, 1)


def _graph_with_node():
    return create_grid(3, 3, seed=0)


def test_draw_default_returns_axes():
    graph = create_grid(5, 5, seed=0)
    ax = draw(graph)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_draw_with_provided_ax_returns_same_ax():
    graph = create_grid(5, 5, seed=0)
    fig, ax = plt.subplots()
    result = draw(graph, ax=ax)
    plt.close(fig)
    assert result is ax


def test_build_rgb_burning_node_renders_as_burning_colour():
    graph = _graph_with_node()
    graph.nodes[_NODE][STATE] = NodeState.BURNING
    assert np.array_equal(_build_rgb(graph, 3, 3)[_NODE], _BURNING)


def test_build_rgb_burned_node_renders_as_burned_colour():
    graph = _graph_with_node()
    graph.nodes[_NODE][STATE] = NodeState.BURNED
    assert np.array_equal(_build_rgb(graph, 3, 3)[_NODE], _BURNED)


def test_build_rgb_treated_node_renders_as_treated_colour():
    graph = _graph_with_node()
    graph.nodes[_NODE][STATE] = NodeState.TREATED
    assert np.array_equal(_build_rgb(graph, 3, 3)[_NODE], _TREATED)


def test_build_rgb_water_terrain_unburned_renders_as_water_colour():
    graph = _graph_with_node()
    graph.nodes[_NODE][TERRAIN] = TerrainType.WATER
    assert np.array_equal(_build_rgb(graph, 3, 3)[_NODE], _WATER)


def test_build_rgb_rock_terrain_unburned_renders_as_rock_colour():
    graph = _graph_with_node()
    graph.nodes[_NODE][TERRAIN] = TerrainType.ROCK
    assert np.array_equal(_build_rgb(graph, 3, 3)[_NODE], _ROCK)


def test_build_rgb_land_unburned_renders_as_fuel_mapped_colour():
    graph = _graph_with_node()
    colour = _build_rgb(graph, 3, 3)[_NODE]
    fixed = [_BURNING, _BURNED, _TREATED, _WATER, _ROCK]
    assert not any(np.array_equal(colour, c) for c in fixed)


def test_build_rgb_burning_state_takes_priority_over_water_terrain():
    graph = _graph_with_node()
    graph.nodes[_NODE][STATE] = NodeState.BURNING
    graph.nodes[_NODE][TERRAIN] = TerrainType.WATER
    assert np.array_equal(_build_rgb(graph, 3, 3)[_NODE], _BURNING)


def test_build_elevation_values_match_node_elevation_attributes():
    graph = create_grid(3, 3, seed=0)
    elevation = _build_elevation(graph, 3, 3)
    assert all(elevation[i, j] == graph.nodes[(i, j)][ELEVATION] for i, j in graph.nodes)


def _snapshots(seed=0, steps=3):
    graph = create_grid(5, 5, seed=seed)
    set_wind(graph, speed=20.0, direction=0.0)
    set_fuel_moisture(graph, moisture=0.1)
    node = (2, 2)
    graph.nodes[node][STATE] = NodeState.BURNING
    graph.nodes[node][BURN_TIMER] = max(1, math.ceil(graph.nodes[node][FUEL] * MAX_BURN_STEPS))
    rng = np.random.default_rng(seed)
    frames = [copy.deepcopy(graph)]
    for _ in range(steps - 1):
        spread_step(graph, rng)
        frames.append(copy.deepcopy(graph))
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
    graph = create_grid(3, 3, seed=0)
    out = tmp_path / "out.gif"
    animate([graph], path=str(out))
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# render_heatmap
# ---------------------------------------------------------------------------


def _heatmap_graph():
    g = create_grid(5, 5, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    return g


def test_render_heatmap_returns_axes():
    g = _heatmap_graph()
    ax = render_heatmap(g, lambda graph, node: 1.0)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_render_heatmap_saves_file(tmp_path):
    g = _heatmap_graph()
    out = tmp_path / "heatmap.png"
    render_heatmap(g, lambda graph, node: 1.0, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_heatmap_handles_all_nonfinite_scores():
    g = _heatmap_graph()
    ax = render_heatmap(g, lambda graph, node: float("-inf"))
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_render_heatmap_handles_uniform_scores():
    g = _heatmap_graph()
    ax = render_heatmap(g, lambda graph, node: 0.5)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_render_heatmap_runs_all_required_precomputes():
    from wildfireGP.features import (
        reachable_unburned_area,
        total_burned,
        total_burning,
        total_treated,
        total_unburned,
    )

    g = _heatmap_graph()

    def strategy_using_all_precomputed_features(graph, node):
        return (
            reachable_unburned_area(graph, node)
            + total_burned(graph)
            + total_burning(graph)
            + total_treated(graph)
            + total_unburned(graph)
        )

    ax = render_heatmap(g, strategy_using_all_precomputed_features)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# animate_heatmap
# ---------------------------------------------------------------------------


def test_animate_heatmap_creates_gif(tmp_path):
    out = tmp_path / "heatmap.gif"
    animate_heatmap(_snapshots(), lambda graph, node: 1.0, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_heatmap_single_frame(tmp_path):
    graph = _heatmap_graph()
    out = tmp_path / "out.gif"
    animate_heatmap([graph], lambda g, n: 1.0, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_heatmap_handles_all_nonfinite_scores(tmp_path):
    out = tmp_path / "out.gif"
    animate_heatmap(_snapshots(), lambda graph, node: float("-inf"), path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_heatmap_runs_all_required_precomputes(tmp_path):
    from wildfireGP.features import (
        reachable_unburned_area,
        total_burned,
        total_burning,
        total_treated,
        total_unburned,
    )

    def strategy_using_all_precomputed_features(graph, node):
        return (
            reachable_unburned_area(graph, node)
            + total_burned(graph)
            + total_burning(graph)
            + total_treated(graph)
            + total_unburned(graph)
        )

    out = tmp_path / "heatmap.gif"
    animate_heatmap(_snapshots(), strategy_using_all_precomputed_features, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_heatmap_global_normalisation_uses_all_frames(tmp_path):
    scores = iter(range(100))

    def incrementing(graph, node):
        return float(next(scores, 0))

    out = tmp_path / "out.gif"
    animate_heatmap(_snapshots(steps=3), incrementing, path=str(out))
    assert out.exists() and out.stat().st_size > 0
