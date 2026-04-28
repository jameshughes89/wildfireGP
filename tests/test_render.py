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


def test_draw_contours_use_default_origin():
    graph = create_grid(5, 5, seed=0)
    fig, ax = plt.subplots()
    captured: dict[str, object] = {}

    def fake_contour(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return None

    original_contour = ax.contour
    ax.contour = fake_contour
    try:
        draw(graph, ax=ax)
    finally:
        ax.contour = original_contour
        plt.close(fig)

    assert "origin" not in captured
    expected = _build_elevation(graph, 5, 5)
    assert np.array_equal(captured["args"][0], expected)
    assert captured["levels"] == 10
    assert captured["colors"] == "white"
    assert captured["alpha"] == 0.5
    assert captured["linewidths"] == 1.0
