import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_timeseries import _plot, collect_series, main
from wildfireGP.network import (
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.strategies import no_treatment


@pytest.fixture()
def small_graph():
    g = create_grid(7, 7, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    return g


@pytest.fixture()
def ignition(small_graph):
    return select_ignition_node(small_graph, np.random.default_rng(0))


def _make_series(n_steps: int = 5) -> dict:
    arr = np.linspace(0, 10, n_steps).reshape(1, -1).repeat(3, axis=0)
    return {"burning": arr, "burned": arr * 2}


# ---------------------------------------------------------------------------
# collect_series
# ---------------------------------------------------------------------------


def test_collect_series_burning_shape(small_graph, ignition):
    runs = 3
    data = collect_series(no_treatment, small_graph, ignition, 0, 10, runs, base_seed=0)
    assert data["burning"].shape[0] == runs


def test_collect_series_burned_shape(small_graph, ignition):
    runs = 3
    data = collect_series(no_treatment, small_graph, ignition, 0, 10, runs, base_seed=0)
    assert data["burned"].shape[0] == runs


def test_collect_series_rows_equal_length(small_graph, ignition):
    data = collect_series(no_treatment, small_graph, ignition, 0, 10, runs=3, base_seed=0)
    assert data["burning"].shape == data["burned"].shape


def test_collect_series_burned_non_negative(small_graph, ignition):
    data = collect_series(no_treatment, small_graph, ignition, 0, 10, runs=2, base_seed=0)
    assert (data["burned"] >= 0).all()


def test_collect_series_burning_non_negative(small_graph, ignition):
    data = collect_series(no_treatment, small_graph, ignition, 0, 10, runs=2, base_seed=0)
    assert (data["burning"] >= 0).all()


# ---------------------------------------------------------------------------
# _plot
# ---------------------------------------------------------------------------


def test_plot_returns_figure():
    fig = _plot({"no_treatment": _make_series()})
    plt.close(fig)
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_creates_output_file(tmp_path):
    out = tmp_path / "ts.png"
    main(
        [
            "--strategy",
            "no_treatment",
            "--rows",
            "5",
            "--cols",
            "5",
            "--runs",
            "2",
            "--max-steps",
            "5",
            "--output",
            str(out),
        ]
    )
    assert out.exists()


def test_main_default_output_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["--strategy", "no_treatment", "--rows", "5", "--cols", "5", "--runs", "2", "--max-steps", "5"])
    assert (tmp_path / "sim_timeseries.png").exists()


def test_main_unknown_strategy_raises():
    with pytest.raises(SystemExit):
        main(["--strategy", "does_not_exist"])


def test_main_no_strategies_uses_all_builtins(tmp_path):
    out = tmp_path / "ts.png"
    main(["--rows", "5", "--cols", "5", "--runs", "2", "--max-steps", "5", "--output", str(out)])
    assert out.exists()
