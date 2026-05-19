"""
Plot per-node treatment and burn frequency heatmaps for a GP or baseline strategy.

Runs N independent simulations on the same landscape and accumulates, for each node:
    - treatment frequency: fraction of runs in which the node was treated at any point
    - burn frequency: fraction of runs in which the node burned

Two heatmaps are plotted side by side. Comparing them reveals whether the strategy places
treatments ahead of the fire (treatment-heavy nodes in low-burn areas) or reactively
(treatment and burn patterns overlap).

Usage
-----
    python -m scripts.plot_treatment_heatmap (--strategy NAME | --hof PATH | --expr STRING)
                                             [--results-dir PATH]
                                             [--runs INT] [--output PATH]
                                             [--seed INT] [--rows INT] [--cols INT]
                                             [--treatments INT] [--max-steps INT]
                                             [--intervention-delay INT]
                                             [--wind-speed FLOAT] [--wind-direction FLOAT]
                                             [--moisture FLOAT]

    Exactly one of --strategy, --hof, or --expr must be supplied.

    --strategy NAME  Name of a builtin baseline from strategies.py.
    --hof PATH       Path to a .dill file produced by run_gp.py.
    --expr STRING    Expression string from batch_evaluate.py CSV output; requires --results-dir.

Examples
--------
    python -m scripts.plot_treatment_heatmap --strategy score_head_fire --runs 50
    python -m scripts.plot_treatment_heatmap --hof results/run/hof_0.dill --runs 50
    python -m scripts.plot_treatment_heatmap --results-dir results/run --expr "min(fuel_level, ...)" --runs 50
"""

import argparse
import copy
import logging
import pathlib
import sys

import dill
import matplotlib.pyplot as plt
import numpy as np

from scripts.cli import add_landscape_args, load_candidate_by_expr
from wildfireGP.evaluate import DEFAULT_INTERVENTION_DELAY, init_ignition, simulate
from wildfireGP.network import (
    STATE,
    NodeState,
    create_grid,
    select_ignition_cluster,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.strategies import ALL_STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_STRATEGY_MAP = {f.__name__: f for f in ALL_STRATEGIES}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    func, label = _load_strategy(args)

    rng = np.random.default_rng(args.seed)
    log.info("Building landscape (%dx%d, seed=%s)", args.rows, args.cols, args.seed)
    graph = create_grid(args.rows, args.cols, seed=args.seed)
    set_wind(graph, speed=args.wind_speed, direction=args.wind_direction)
    set_fuel_moisture(graph, moisture=args.moisture)
    ignition_nodes = select_ignition_cluster(graph, rng, size=args.ignition_cluster_size)
    log.info("Ignition cluster (%d nodes): %s  strategy: %s", len(ignition_nodes), ignition_nodes, label)

    log.info("Running %d simulations ...", args.runs)
    treatment_freq, burn_freq = collect_spatial_data(
        func, graph, ignition_nodes, args.treatments, args.max_steps, args.runs, args.seed, args.intervention_delay
    )

    output = pathlib.Path(args.output) if args.output else pathlib.Path("treatment_heatmap.png")
    fig = _plot(treatment_freq, burn_freq, label)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output)


def collect_spatial_data(
    func,
    graph,
    ignition_nodes: list[tuple],
    treatments_per_step: int,
    max_steps: int,
    runs: int,
    base_seed: int | None,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run `runs` simulations and return per-node treatment and burn frequency arrays.

    :return: (treatment_freq, burn_freq) each an ndarray of shape (rows, cols) with values in [0, 1].
    """
    nodes = list(graph.nodes)
    rows = max(r for r, _ in nodes) + 1
    cols = max(c for _, c in nodes) + 1
    treatment_count = np.zeros((rows, cols), dtype=int)
    burn_count = np.zeros((rows, cols), dtype=int)

    for i in range(runs):
        seed = None if base_seed is None else base_seed + i
        treated, burned = _run_and_collect(
            graph, ignition_nodes, func, treatments_per_step, max_steps, np.random.default_rng(seed), intervention_delay
        )
        for r, c in treated:
            treatment_count[r, c] += 1
        for r, c in burned:
            burn_count[r, c] += 1

    return treatment_count / runs, burn_count / runs


def _run_and_collect(
    graph,
    ignition_nodes: list[tuple],
    func,
    treatments_per_step: int,
    max_steps: int,
    rng,
    intervention_delay: int,
) -> tuple[set, set]:
    g = copy.deepcopy(graph)
    init_ignition(g, ignition_nodes)
    for _step, _ in simulate(g, func, treatments_per_step, max_steps, rng, intervention_delay):
        pass
    treated_nodes = {n for n in g.nodes if g.nodes[n][STATE] == NodeState.TREATED}
    burned_nodes = {n for n in g.nodes if g.nodes[n][STATE] in (NodeState.BURNED, NodeState.BURNING)}
    return treated_nodes, burned_nodes


def _plot(treatment_freq: np.ndarray, burn_freq: np.ndarray, label: str) -> plt.Figure:
    fig, (ax_treat, ax_burn) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Spatial frequency — {label[:60]}", fontsize=11)

    im_t = ax_treat.imshow(treatment_freq, vmin=0, vmax=1, cmap="Blues", origin="upper")
    fig.colorbar(im_t, ax=ax_treat, label="treatment frequency")
    ax_treat.set_title("Treatment frequency")
    ax_treat.set_xlabel("col")
    ax_treat.set_ylabel("row")

    im_b = ax_burn.imshow(burn_freq, vmin=0, vmax=1, cmap="YlOrRd", origin="upper")
    fig.colorbar(im_b, ax=ax_burn, label="burn frequency")
    ax_burn.set_title("Burn frequency")
    ax_burn.set_xlabel("col")
    ax_burn.set_ylabel("row")

    fig.tight_layout()
    return fig


def _load_strategy(args: argparse.Namespace) -> tuple:
    if args.hof:
        path = pathlib.Path(args.hof)
        with open(path, "rb") as f:
            func = dill.load(f)
        return func, path.stem
    if args.expr:
        if not args.results_dir:
            raise SystemExit("--results-dir is required when using --expr.")
        func = load_candidate_by_expr(args.results_dir, args.expr)
        return func, args.expr[:60]
    name = args.strategy
    if name not in _STRATEGY_MAP:
        available = ", ".join(sorted(_STRATEGY_MAP))
        raise SystemExit(f"Unknown strategy '{name}'. Available: {available}")
    return _STRATEGY_MAP[name], name


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-node treatment and burn frequency heatmaps.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--strategy", type=str, help="Builtin strategy name.")
    source.add_argument("--hof", type=pathlib.Path, help="Path to a .dill HOF file.")
    source.add_argument("--expr", type=str, help="Load a candidate by expression string; requires --results-dir.")
    parser.add_argument("--results-dir", type=pathlib.Path, default=None)
    add_landscape_args(parser)
    parser.add_argument("--ignition-cluster-size", type=int, default=3, help="Number of nodes to ignite at t=0.")
    parser.add_argument("--runs", type=int, default=50, help="Simulations to accumulate (default 50).")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
