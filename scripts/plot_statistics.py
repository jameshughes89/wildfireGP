"""
Plot fitness and diagnostic statistics from a GP run's stats.json.

Produces a 2x2 figure:
    top-left:     total_burned (min and avg) over generations --- the fitness objective
    top-right:    peak_burning (min and avg) over generations --- diagnostic
    bottom-left:  tree size in nodes (avg and max) over generations --- bloat indicator
    bottom-right: tree height (avg and max) over generations --- bloat indicator

Usage
-----
    python -m scripts.plot_statistics --results-dir PATH [--output PATH]
    python -m scripts.plot_statistics --stats PATH [--output PATH]

    Exactly one of --results-dir or --stats must be supplied.

    --results-dir PATH  Directory produced by run_gp.py; reads PATH/stats.json and saves
                        the plot as PATH/stats_plot.png by default.
    --stats PATH        Path to a stats.json file directly.
    --output PATH       Override the output file path.

Examples
--------
    python -m scripts.plot_statistics --results-dir results/2026-05-08_10-00-00
    python -m scripts.plot_statistics --stats results/2026-05-08_10-00-00/stats.json --output run_plot.png
"""

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    stats_path, default_output = _resolve_paths(args)

    rows = json.loads(stats_path.read_text())
    if not rows:
        raise SystemExit("stats.json is empty --- nothing to plot.")

    output = pathlib.Path(args.output) if args.output else default_output
    fig = _plot(rows)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def _plot(rows: list[dict]) -> plt.Figure:
    gens = [r["gen"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GP run statistics", fontsize=13)

    ax = axes[0, 0]
    ax.plot(gens, [r["fitness_min"] for r in rows], label="min")
    ax.plot(gens, [r["fitness_avg"] for r in rows], label="avg", linestyle="--")
    ax.set_title("total_burned (fitness)")
    ax.set_xlabel("generation")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(gens, [r["peak_min"] for r in rows], label="min")
    ax.plot(gens, [r["peak_avg"] for r in rows], label="avg", linestyle="--")
    ax.set_title("peak_burning (diagnostic)")
    ax.set_xlabel("generation")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(gens, [r["size_avg"] for r in rows], label="avg")
    ax.plot(gens, [r["size_max"] for r in rows], label="max", linestyle="--")
    ax.set_title("tree size (nodes)")
    ax.set_xlabel("generation")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(gens, [r["height_avg"] for r in rows], label="avg")
    ax.plot(gens, [r["height_max"] for r in rows], label="max", linestyle="--")
    ax.set_title("tree height")
    ax.set_xlabel("generation")
    ax.legend()

    fig.tight_layout()
    return fig


def _resolve_paths(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    if args.results_dir:
        stats_path = args.results_dir / "stats.json"
        default_output = args.results_dir / "stats_plot.png"
    else:
        stats_path = pathlib.Path(args.stats)
        default_output = stats_path.parent / "stats_plot.png"
    if not stats_path.exists():
        raise SystemExit(f"stats.json not found: {stats_path}")
    return stats_path, default_output


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GP run statistics from stats.json.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-dir", type=pathlib.Path)
    source.add_argument("--stats", type=pathlib.Path)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
