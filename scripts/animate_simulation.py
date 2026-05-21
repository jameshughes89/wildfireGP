"""
Run a single simulation with a chosen strategy and save a GIF.

Frames are captured after each spread step and passed to render.animate() which writes the GIF.

Usage
-----
    python -m scripts.animate_simulation [--output PATH] [--strategy NAME] [--hof PATH] [--expr STRING]
                              [--results-dir PATH]
                              [--seed INT] [--rows INT] [--cols INT]
                              [--treatments INT] [--max-steps INT] [--intervention-delay INT]
                              [--wind-speed FLOAT] [--wind-direction FLOAT] [--moisture FLOAT]
                              [--fps INT]

    Exactly one of --strategy, --hof, or --expr must be supplied.

    --strategy NAME  Name of a builtin baseline from strategies.py, e.g. score_head_fire.
    --hof PATH       Path to a .dill file produced by run_gp.py.
    --expr STRING    Expression string of a specific GP candidate; requires --results-dir. Copy the expr value from
                     the batch_evaluate.py CSV output.

Examples
--------
    python -m scripts.animate_simulation --strategy score_head_fire --output head_fire.gif
    python -m scripts.animate_simulation --hof results/2026-05-06_14-32-00/hof_0.dill --output gp_best.gif
    python -m scripts.animate_simulation --results-dir results/2026-05-06_14-32-00 --expr "min(fuel_level, ...)"
"""

import argparse
import logging
import pathlib
import sys

import dill
import numpy as np

from scripts.cli import add_landscape_args, load_candidate_by_expr
from wildfireGP.evaluate import DEFAULT_INTERVENTION_DELAY, init_ignition, simulate
from wildfireGP.network import (
    create_grid,
    select_ignition_cluster,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.render import animate
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

    init_ignition(graph, ignition_nodes)
    snapshots = _run_simulation(graph, func, args.treatments, args.max_steps, rng, args.intervention_delay)
    log.info("Simulation complete: %d frames", len(snapshots))

    output = pathlib.Path(args.output)
    log.info("Saving GIF to %s", output)
    animate(snapshots, path=str(output), fps=args.fps)
    log.info("Done.")


def _run_simulation(
    state,
    func,
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> list:
    s = state.copy()
    snapshots = [s.copy()]
    for _step, _ in simulate(s, func, treatments_per_step, max_steps, rng, intervention_delay):
        snapshots.append(s.copy())
    return snapshots


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
    parser = argparse.ArgumentParser(description="Animate a wildfire simulation with a chosen strategy.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--strategy", type=str, help="Builtin strategy name.")
    source.add_argument("--hof", type=pathlib.Path, help="Path to a .dill HOF file.")
    source.add_argument("--expr", type=str, help="Load a candidate by expression string; requires --results-dir.")
    parser.add_argument("--results-dir", type=pathlib.Path, default=None)
    add_landscape_args(parser)
    parser.add_argument("--ignition-cluster-size", type=int, default=3, help="Number of nodes to ignite at t=0.")
    parser.add_argument("--output", type=str, default="simulation.gif")
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
