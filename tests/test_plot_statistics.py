import json

import matplotlib.pyplot as plt
import pytest

from scripts.plot_statistics import _plot, main


def _make_stats(n_gens: int = 3) -> list[dict]:
    return [
        {
            "gen": g,
            "fitness_avg": 20.0 - g,
            "fitness_min": 15.0 - g,
            "fitness_max": 25.0 - g,
            "fitness_std": 2.0,
            "peak_avg": 10.0 - g * 0.5,
            "peak_min": 8.0 - g * 0.5,
            "peak_max": 12.0,
            "peak_std": 1.0,
            "size_avg": 7.0 + g * 0.2,
            "size_min": 3.0,
            "size_max": 15.0 + g,
            "size_std": 2.0,
            "height_avg": 3.0 + g * 0.1,
            "height_min": 1.0,
            "height_max": 5.0,
            "height_std": 0.5,
        }
        for g in range(n_gens)
    ]


def test_plot_returns_figure(tmp_path):
    fig = _plot(_make_stats())
    plt.close(fig)
    assert fig is not None


def test_plot_has_six_subplots(tmp_path):
    fig = _plot(_make_stats())
    assert len(fig.axes) >= 6
    plt.close(fig)


def test_plot_stats_creates_output_file(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(_make_stats()))
    main(["--stats", str(stats_path), "--output", str(tmp_path / "out.png")])
    assert (tmp_path / "out.png").exists()


def test_plot_stats_via_results_dir(tmp_path):
    (tmp_path / "stats.json").write_text(json.dumps(_make_stats()))
    main(["--results-dir", str(tmp_path), "--output", str(tmp_path / "out.png")])
    assert (tmp_path / "out.png").exists()


def test_plot_stats_missing_file_raises(tmp_path):
    with pytest.raises(SystemExit):
        main(["--stats", str(tmp_path / "nonexistent.json")])


def test_plot_stats_empty_file_raises(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("[]")
    with pytest.raises(SystemExit):
        main(["--stats", str(stats_path)])


def test_plot_stats_single_generation(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(_make_stats(n_gens=1)))
    main(["--stats", str(stats_path), "--output", str(tmp_path / "out.png")])
    assert (tmp_path / "out.png").exists()


def test_plot_stats_default_output_path(tmp_path):
    (tmp_path / "stats.json").write_text(json.dumps(_make_stats()))
    main(["--results-dir", str(tmp_path)])
    assert (tmp_path / "stats_plot.png").exists()


def test_plot_stats_missing_key_raises(tmp_path):
    partial = [{"gen": 0, "fitness_avg": 10.0}]
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(partial))
    with pytest.raises(KeyError):
        main(["--stats", str(stats_path), "--output", str(tmp_path / "out.png")])
