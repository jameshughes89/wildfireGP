"""
Plot simulation state time-series for one or more strategies.

For each strategy, runs N independent simulations on the same landscape and ignition point. The mean and +/-1 std band
of active burning nodes and cumulative burned nodes are plotted over timesteps.

Usage
-----
    python -m scripts.plot_timeseries [--strategy NAME [NAME ...]] [--hof PATH [PATH ...]]
                               [--results-dir PATH] [--expr STRING]
                               [--seed INT] [--rows INT] [--cols INT]
                               [--treatments INT] [--max-steps INT] [--intervention-delay INT]
                               [--wind-speed FLOAT] [--wind-direction FLOAT] [--moisture FLOAT]
                               [--runs INT] [--output PATH]

    At least one of --strategy or --hof must be given. If neither is supplied, all builtin strategies are plotted.

    --expr STRING    Expression string of a specific GP candidate to load from --results-dir (required alongside
                     --expr). Copy the expr value from batch_evaluate.py CSV output. Mutually exclusive with --hof.

Examples
--------
    python -m scripts.plot_timeseries --strategy random_score score_by_fire_proximity
    python -m scripts.plot_timeseries --hof results/run/hof_0.dill --strategy random_score --runs 30
    python -m scripts.plot_timeseries --results-dir results/run --expr "min(fuel_level, ...)" --strategy random_score
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
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.strategies import ALL_STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_STRATEGY_MAP = {f.__name__: f for f in ALL_STRATEGIES}


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
    log.info("Running %d strategies, %d runs each", len(strategies), args.runs)

    series_data = {}
    for name, func in strategies:
        log.info("  %s ...", name)
        series_data[name] = collect_series(
            func, graph, ignition, args.treatments, args.max_steps, args.runs, args.seed, args.intervention_delay
        )

    output = pathlib.Path(args.output) if args.output else pathlib.Path("sim_timeseries.png")
    fig = _plot(series_data)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output)


def collect_series(
    func,
    graph,
    ignition: tuple,
    treatments_per_step: int,
    max_steps: int,
    runs: int,
    base_seed: int | None,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> dict[str, np.ndarray]:
    """
    Run `runs` simulations and return per-step count arrays.

    :return: Dict with keys "burning" and "burned", each an ndarray of shape (runs, max_steps+1).
        Shorter runs are padded with their final value so all rows have equal length.
    """
    all_burning: list[list[int]] = []
    all_burned: list[list[int]] = []

    for i in range(runs):
        seed = None if base_seed is None else base_seed + i
        burning, burned = _run_simulation(
            graph, ignition, func, treatments_per_step, max_steps, np.random.default_rng(seed), intervention_delay
        )
        all_burning.append(burning)
        all_burned.append(burned)

    max_len = max(len(r) for r in all_burning)

    def _pad(runs_list: list[list[int]]) -> np.ndarray:
        return np.array([r + [r[-1]] * (max_len - len(r)) for r in runs_list], dtype=float)

    return {"burning": _pad(all_burning), "burned": _pad(all_burned)}


def _run_simulation(
    graph, ignition: tuple, func, treatments_per_step: int, max_steps: int, rng, intervention_delay: int
) -> tuple[list[int], list[int]]:
    g = copy.deepcopy(graph)
    init_ignition(g, [ignition])
    burning = [sum(1 for n in g.nodes if g.nodes[n][STATE] == NodeState.BURNING)]
    burned = [sum(1 for n in g.nodes if g.nodes[n][STATE] == NodeState.BURNED)]
    for _step, _ in simulate(g, func, treatments_per_step, max_steps, rng, intervention_delay):
        burning.append(sum(1 for n in g.nodes if g.nodes[n][STATE] == NodeState.BURNING))
        burned.append(sum(1 for n in g.nodes if g.nodes[n][STATE] == NodeState.BURNED))
    return burning, burned


def _plot(series_data: dict[str, dict]) -> plt.Figure:
    fig, (ax_burning, ax_burned) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Simulation time-series (mean ± 1 std)", fontsize=13)

    for name, data in series_data.items():
        for ax, key, ylabel in [
            (ax_burning, "burning", "active burning nodes"),
            (ax_burned, "burned", "cumulative burned nodes"),
        ]:
            arr = data[key]
            steps = np.arange(arr.shape[1])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            (line,) = ax.plot(steps, mean, label=name)
            ax.fill_between(steps, np.maximum(0, mean - std), mean + std, alpha=0.2, color=line.get_color())

    for ax, title in [(ax_burning, "Active burning nodes"), (ax_burned, "Cumulative burned nodes")]:
        ax.set_xlabel("step")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def _load_strategies(args: argparse.Namespace) -> list[tuple[str, object]]:
    strategies = []

    for name in args.strategy or []:
        if name not in _STRATEGY_MAP:
            available = ", ".join(sorted(_STRATEGY_MAP))
            raise SystemExit(f"Unknown strategy '{name}'. Available: {available}")
        strategies.append((name, _STRATEGY_MAP[name]))

    if getattr(args, "expr", None):
        if not getattr(args, "results_dir", None):
            raise SystemExit("--results-dir is required when using --expr.")
        func = load_candidate_by_expr(args.results_dir, args.expr)
        strategies.append((args.expr[:60], func))
    else:
        for path in args.hof or []:
            path = pathlib.Path(path)
            with open(path, "rb") as f:
                func = dill.load(f)
            strategies.append((path.stem, func))

    if not strategies:
        strategies = [(f.__name__, f) for f in ALL_STRATEGIES]

    return strategies


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simulation state time-series for strategies.")
    parser.add_argument("--strategy", type=str, nargs="+", help="Builtin strategy names.")
    gp_source = parser.add_mutually_exclusive_group()
    gp_source.add_argument("--hof", type=pathlib.Path, nargs="+", help="Paths to .dill HOF files.")
    gp_source.add_argument("--expr", type=str, default=None, help="Load a candidate by expression string.")
    parser.add_argument("--results-dir", type=pathlib.Path, default=None)
    add_landscape_args(parser)
    parser.add_argument("--runs", type=int, default=20, help="Stochastic runs per strategy.")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
