import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_comparison import _plot, main


def _make_results(n_strategies: int = 3, runs: int = 5) -> dict:
    rng = np.random.default_rng(0)
    return {
        f"strategy_{i}": (rng.integers(10, 50, size=runs).astype(float), rng.integers(5, 20, size=runs).astype(float))
        for i in range(n_strategies)
    }


# ---------------------------------------------------------------------------
# _plot
# ---------------------------------------------------------------------------


def test_plot_returns_figure():
    fig = _plot(_make_results())
    plt.close(fig)
    assert isinstance(fig, plt.Figure)


def test_plot_strategy_names_in_figure():
    results = {"no_treatment": (np.array([50.0, 60.0]), np.array([10.0, 12.0]))}
    fig = _plot(results)
    labels = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels()]
    plt.close(fig)
    assert any("no_treatment" in label for label in labels)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_creates_output_file(tmp_path):
    out = tmp_path / "comp.png"
    main(["--rows", "5", "--cols", "5", "--runs", "3", "--max-steps", "10", "--output", str(out)])
    assert out.exists()


def test_main_default_output_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["--rows", "5", "--cols", "5", "--runs", "3", "--max-steps", "10"])
    assert (tmp_path / "strategy_comparison.png").exists()


def test_main_with_results_dir_loads_dill(tmp_path):
    import dill

    from wildfireGP.strategies import no_treatment

    (tmp_path / "hof_0.dill").write_bytes(dill.dumps(no_treatment))
    out = tmp_path / "comp.png"
    main(
        [
            "--results-dir",
            str(tmp_path),
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


def test_main_expr_loads_candidate(tmp_path):
    import dill

    from wildfireGP.strategies import no_treatment

    expr = "test_expr"
    with open(tmp_path / "final_population.dill", "wb") as f:
        dill.dump([(expr, no_treatment)], f)
    out = tmp_path / "comp.png"
    main(
        [
            "--results-dir",
            str(tmp_path),
            "--expr",
            expr,
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
