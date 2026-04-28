import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from wildfireGP.network import create_grid
from wildfireGP.render import draw, draw_contours, draw_elevation, draw_hillshaded


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


def test_draw_hillshaded_returns_axes():
    graph = create_grid(5, 5, seed=0)
    ax = draw_hillshaded(graph)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_draw_contours_returns_axes():
    graph = create_grid(5, 5, seed=0)
    ax = draw_contours(graph)
    plt.close("all")
    assert isinstance(ax, plt.Axes)


def test_draw_elevation_returns_axes():
    graph = create_grid(5, 5, seed=0)
    ax = draw_elevation(graph)
    plt.close("all")
    assert isinstance(ax, plt.Axes)
