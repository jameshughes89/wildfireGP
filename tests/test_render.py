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
    draw,
)
from wildfireGP.spread import MAX_BURN_STEPS

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


def test_build_rgb_shape_is_rows_by_cols_by_3():
    graph = create_grid(4, 7, seed=0)
    assert _build_rgb(graph, 4, 7).shape == (4, 7, 3)


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


def test_build_rgb_burning_state_takes_priority_over_rock_terrain():
    graph = _graph_with_node()
    graph.nodes[_NODE][STATE] = NodeState.BURNING
    graph.nodes[_NODE][TERRAIN] = TerrainType.ROCK
    assert np.array_equal(_build_rgb(graph, 3, 3)[_NODE], _BURNING)


def test_build_elevation_shape_is_rows_by_cols():
    graph = create_grid(4, 7, seed=0)
    assert _build_elevation(graph, 4, 7).shape == (4, 7)


def test_build_elevation_values_match_node_elevation_attributes():
    graph = create_grid(3, 3, seed=0)
    elevation = _build_elevation(graph, 3, 3)
    assert all(elevation[i, j] == graph.nodes[(i, j)][ELEVATION] for i, j in graph.nodes)


def _burning_graph(seed=0):
    graph = create_grid(5, 5, seed=seed)
    set_wind(graph, speed=20.0, direction=0.0)
    set_fuel_moisture(graph, moisture=0.1)
    node = (2, 2)
    graph.nodes[node][STATE] = NodeState.BURNING
    graph.nodes[node][BURN_TIMER] = max(1, math.ceil(graph.nodes[node][FUEL] * MAX_BURN_STEPS))
    return graph


def test_animate_creates_gif(tmp_path):
    out = tmp_path / "out.gif"
    rng = np.random.default_rng(0)
    animate(_burning_graph(), steps=3, rng=rng, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_does_not_leave_open_figures(tmp_path):
    before = len(plt.get_fignums())
    rng = np.random.default_rng(0)
    animate(_burning_graph(), steps=3, rng=rng, path=str(tmp_path / "out.gif"))
    assert len(plt.get_fignums()) == before


def test_animate_advances_simulation(tmp_path):
    graph = _burning_graph(seed=1)
    rng = np.random.default_rng(1)
    animate(graph, steps=5, rng=rng, path=str(tmp_path / "out.gif"))
    states = [graph.nodes[n][STATE] for n in graph.nodes]
    assert NodeState.BURNED in states or NodeState.BURNING in states


def test_animate_stops_early_when_fire_extinguished(tmp_path):
    graph = create_grid(3, 3, seed=0)
    set_wind(graph, speed=0.0, direction=0.0)
    set_fuel_moisture(graph, moisture=1.0)
    node = (1, 1)
    graph.nodes[node][STATE] = NodeState.BURNING
    graph.nodes[node][BURN_TIMER] = 1
    rng = np.random.default_rng(0)
    animate(graph, steps=20, rng=rng, path=str(tmp_path / "out.gif"))
    assert (tmp_path / "out.gif").exists()
