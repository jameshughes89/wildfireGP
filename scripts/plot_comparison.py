"""
Plot strategy comparison as box plots.

Each strategy is evaluated over --runs independent simulations on the same landscape. Box plots of total_burned and
peak_burning distributions are shown side-by-side, ranked by median total_burned.

Usage
-----
    python -m scripts.plot_comparison [--results-dir PATH] [--hof PATH [PATH ...]]
                                      [--seed INT] [--rows INT] [--cols INT]
                                      [--treatments INT] [--max-steps INT]
                                      [--runs INT] [--output PATH]

    If --results-dir is given, all .dill files in that directory are loaded automatically alongside the builtin
    baselines. If neither --hof nor --results-dir is given, only baselines are compared.

Examples
--------
    python -m scripts.plot_comparison
    python -m scripts.plot_comparison --results-dir results/2026-05-08_10-00-00
    python -m scripts.plot_comparison --hof results/run/hof_0.dill --runs 50
"""

import argparse
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

from scripts.cli import add_landscape_args
from scripts.compare_strategies import _load_strategies, _run_strategy
from wildfireGP.network import (
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    rng = np.random.default_rng(args.seed)
    log.info("Building landscape (%dx%d, seed=%s)", args.rows, args.cols, args.seed)
    graph = create_grid(args.rows, args.cols, seed=args.seed)
    set_wind(graph, speed=args.wind_speed, direction=args.wind_direction)
    set_fuel_moisture(graph, moisture=args.moisture)
    ignition = select_ignition_node(graph, rng)
    log.info("Ignition node: %s", ignition)

    strategies = _load_strategies(args)
    log.info("Evaluating %d strategies over %d runs each", len(strategies), args.runs)

    results = {}
    for name, func in strategies:
        log.info("  %s ...", name)
        burned, peak = _run_strategy(
            func, graph, [ignition], args.treatments, args.max_steps, args.runs, args.seed, args.intervention_delay
        )
        results[name] = (burned, peak)

    results = dict(sorted(results.items(), key=lambda kv: np.median(kv[1][0])))

    output = pathlib.Path(args.output) if args.output else pathlib.Path("strategy_comparison.png")
    fig = _plot(results)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output)


def _plot(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> plt.Figure:
    names = list(results.keys())
    burned_data = [results[n][0] for n in names]
    peak_data = [results[n][1] for n in names]

    fig, (ax_burned, ax_peak) = plt.subplots(1, 2, figsize=(max(8, len(names) * 1.5), 5))
    fig.suptitle("Strategy comparison (box plots)", fontsize=13)

    for ax, data, title, ylabel in [
        (ax_burned, burned_data, "Total burned nodes", "nodes"),
        (ax_peak, peak_data, "Peak burning nodes", "nodes"),
    ]:
        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Box-plot comparison of strategy outcomes.")
    add_landscape_args(parser)
    parser.add_argument("--runs", type=int, default=30, help="Stochastic runs per strategy.")
    parser.add_argument("--results-dir", type=pathlib.Path, default=None)
    parser.add_argument("--hof", type=pathlib.Path, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
