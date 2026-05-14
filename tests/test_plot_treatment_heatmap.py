"""Tests for scripts/plot_treatment_heatmap.py."""

import dill
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_treatment_heatmap import (
    _load_strategy,
    _plot,
    collect_spatial_data,
    main,
)
from wildfireGP.network import (
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.strategies import random_score, score_by_fire_proximity


@pytest.fixture()
def small_graph():
    g = create_grid(7, 7, seed=0)
    set_wind(g, speed=20.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.1)
    return g


@pytest.fixture()
def ignition(small_graph):
    return select_ignition_node(small_graph, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# collect_spatial_data
# ---------------------------------------------------------------------------


def test_collect_spatial_data_treatment_freq_shape(small_graph, ignition):
    treat, _ = collect_spatial_data(random_score, small_graph, ignition, 2, 10, runs=3, base_seed=0)
    assert treat.shape == (7, 7)


def test_collect_spatial_data_burn_freq_shape(small_graph, ignition):
    _, burn = collect_spatial_data(random_score, small_graph, ignition, 2, 10, runs=3, base_seed=0)
    assert burn.shape == (7, 7)


def test_collect_spatial_data_frequencies_in_unit_interval(small_graph, ignition):
    treat, burn = collect_spatial_data(random_score, small_graph, ignition, 2, 10, runs=3, base_seed=0)
    assert (treat >= 0).all() and (treat <= 1).all()
    assert (burn >= 0).all() and (burn <= 1).all()


def test_collect_spatial_data_treatment_strategy_produces_nonzero_treatments(small_graph, ignition):
    treat, _ = collect_spatial_data(score_by_fire_proximity, small_graph, ignition, 5, 20, runs=5, base_seed=0)
    assert treat.sum() > 0


def test_collect_spatial_data_burn_freq_nonzero(small_graph, ignition):
    _, burn = collect_spatial_data(random_score, small_graph, ignition, 0, 20, runs=3, base_seed=0)
    assert burn.sum() > 0


# ---------------------------------------------------------------------------
# _plot
# ---------------------------------------------------------------------------


def test_plot_returns_figure():
    rng = np.random.default_rng(0)
    fig = _plot(rng.uniform(0, 1, (7, 7)), rng.uniform(0, 1, (7, 7)), "test_strategy")
    plt.close(fig)
    assert isinstance(fig, plt.Figure)


def test_plot_has_at_least_two_axes():
    fig = _plot(np.zeros((5, 5)), np.zeros((5, 5)), "test")
    plt.close(fig)
    assert len(fig.axes) >= 2


# ---------------------------------------------------------------------------
# _load_strategy
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, strategy=None, hof=None, expr=None, results_dir=None):
        self.strategy = strategy
        self.hof = hof
        self.expr = expr
        self.results_dir = results_dir


def test_load_strategy_builtin_returns_callable():
    func, _ = _load_strategy(_FakeArgs(strategy="random_score"))
    assert callable(func)


def test_load_strategy_builtin_label_matches_name():
    _, label = _load_strategy(_FakeArgs(strategy="random_score"))
    assert label == "random_score"


def test_load_strategy_unknown_name_raises():
    with pytest.raises(SystemExit):
        _load_strategy(_FakeArgs(strategy="does_not_exist"))


def test_load_strategy_dill_returns_callable(tmp_path):
    path = tmp_path / "hof_0.dill"
    with open(path, "wb") as f:
        dill.dump(random_score, f)
    func, _ = _load_strategy(_FakeArgs(hof=path))
    assert callable(func)


def test_load_strategy_dill_label_is_stem(tmp_path):
    path = tmp_path / "hof_0.dill"
    with open(path, "wb") as f:
        dill.dump(random_score, f)
    _, label = _load_strategy(_FakeArgs(hof=path))
    assert label == "hof_0"


def test_load_strategy_expr_returns_callable(tmp_path):
    expr = "test_expr"
    with open(tmp_path / "final_population.dill", "wb") as f:
        dill.dump([(expr, random_score)], f)
    func, _ = _load_strategy(_FakeArgs(expr=expr, results_dir=tmp_path))
    assert callable(func)


def test_load_strategy_expr_without_results_dir_raises():
    with pytest.raises(SystemExit):
        _load_strategy(_FakeArgs(expr="test_expr"))


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_creates_output_file(tmp_path):
    out = tmp_path / "heatmap.png"
    main(
        [
            "--strategy",
            "random_score",
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
    main(["--strategy", "random_score", "--rows", "5", "--cols", "5", "--runs", "2", "--max-steps", "5"])
    assert (tmp_path / "treatment_heatmap.png").exists()
