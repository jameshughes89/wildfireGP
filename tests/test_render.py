import matplotlib.pyplot as plt
import numpy as np

from wildfireGP.network import create_grid
from wildfireGP.render import _build_elevation, draw


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
