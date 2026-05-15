"""
Plot strategy comparison as box plots.

Each strategy is evaluated across --landscapes independent landscapes with --runs simulations per landscape. Box plots
of total_burned and peak_burning distributions are shown side-by-side, ranked by median total_burned.

Usage
-----
    python -m scripts.plot_comparison [--results-dir PATH] [--hof PATH [PATH ...]]
                                      [--seed INT] [--rows INT] [--cols INT]
                                      [--treatments INT] [--max-steps INT] [--intervention-delay INT]
                                      [--wind-speed FLOAT] [--wind-direction FLOAT] [--moisture FLOAT]
                                      [--landscapes INT] [--runs INT] [--output PATH]

    If --results-dir is given without --expr, all hof_*.dill files in that directory are loaded automatically alongside
    the builtin baselines. If neither --hof nor --results-dir is given, only baselines are compared.

    --expr STRING  Expression string of a specific GP candidate to load from --results-dir. Copy the expr value from
                   the batch_evaluate.py CSV output. --expr is mutually exclusive with --hof.

Examples
--------
    python -m scripts.plot_comparison
    python -m scripts.plot_comparison --results-dir results/2026-05-08_10-00-00
    python -m scripts.plot_comparison --hof results/run/hof_0.dill --runs 20 --landscapes 5
    python -m scripts.plot_comparison --results-dir results/2026-05-08_10-00-00 --expr "min(fuel_level, ...)"
"""

import argparse
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

from scripts.cli import (
    add_landscape_args,
    add_multi_landscape_args,
    find_valid_ignition,
)
from scripts.compare_strategies import _load_strategies, _run_strategy
from wildfireGP.network import (
    create_grid,
    set_fuel_moisture,
    set_wind,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    root_rng = np.random.default_rng(args.seed)

    land_seeds = [int(root_rng.integers(2**31)) if args.seed is not None else None for _ in range(args.landscapes)]
    run_seed_matrix = [
        [int(root_rng.integers(2**31)) if args.seed is not None else None for _ in range(args.runs)]
        for _ in range(args.landscapes)
    ]

    log.info("Validating ignitions across %d landscapes", args.landscapes)
    landscapes = []
    for land_seed in land_seeds:
        graph = create_grid(args.rows, args.cols, seed=land_seed)
        set_wind(graph, speed=args.wind_speed, direction=args.wind_direction)
        set_fuel_moisture(graph, moisture=args.moisture)
        ignition = find_valid_ignition(
            graph,
            np.random.default_rng(land_seed),
            args.max_steps,
            args.intervention_delay,
            args.min_burned,
            args.max_ignition_tries,
        )
        landscapes.append((land_seed, ignition))

    strategies = _load_strategies(args)
    log.info("Evaluating %d strategies over %d landscapes x %d runs", len(strategies), args.landscapes, args.runs)

    results = {}
    for name, func in strategies:
        log.info("  %s ...", name)
        burned, peak = _run_strategy(
            func,
            landscapes,
            args.treatments,
            args.rows,
            args.cols,
            args.wind_speed,
            args.wind_direction,
            args.moisture,
            args.max_steps,
            run_seed_matrix,
            args.intervention_delay,
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
    add_multi_landscape_args(parser)
    parser.add_argument("--runs", type=int, default=5, help="Simulations per landscape (default 5).")
    parser.add_argument("--results-dir", type=pathlib.Path, default=None)
    gp_source = parser.add_mutually_exclusive_group()
    gp_source.add_argument("--hof", type=pathlib.Path, nargs="+", default=None)
    gp_source.add_argument("--expr", type=str, default=None, help="Load a candidate by expression string.")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
