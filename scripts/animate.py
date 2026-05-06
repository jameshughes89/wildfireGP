"""
Run a single simulation with a chosen strategy and save a GIF.

The simulation loop runs spread_step directly, capturing a deepcopy of the graph after each step. Frames are passed to
render.animate() which writes the GIF.

Usage
-----
    python -m scripts.animate [--output PATH] [--strategy NAME] [--hof PATH]
                              [--seed INT] [--rows INT] [--cols INT]
                              [--treatments INT] [--max-steps INT] [--fps INT]

    Exactly one of --strategy or --hof must be supplied.

    --strategy NAME  Name of a builtin baseline from strategies.py, e.g. score_head_fire.
    --hof PATH       Path to a .dill file produced by run_gp.py.

Examples
--------
    python -m scripts.animate --strategy score_head_fire --output head_fire.gif
    python -m scripts.animate --hof results/2026-05-06_14-32-00/hof_0.dill --output gp_best.gif
"""

import argparse
import copy
import logging
import math
import pathlib
import sys

import dill
import numpy as np

from scripts.cli import add_landscape_args
from wildfireGP.features import (
    TREATMENTS_REMAINING,
    precompute_burnable_fire_map,
    precompute_fire_map,
)
from wildfireGP.network import (
    BURN_TIMER,
    FUEL,
    STATE,
    NodeState,
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.render import animate
from wildfireGP.spread import MAX_BURN_STEPS, spread_step
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
    set_wind(graph, speed=20.0, direction=0.0)
    set_fuel_moisture(graph, moisture=0.2)

    ignition = select_ignition_node(graph, rng)
    log.info("Ignition node: %s  strategy: %s", ignition, label)

    graph.nodes[ignition][STATE] = NodeState.BURNING
    graph.nodes[ignition][BURN_TIMER] = max(1, math.ceil(graph.nodes[ignition][FUEL] * MAX_BURN_STEPS))

    snapshots = _run_simulation(graph, func, args.treatments, args.max_steps, rng)
    log.info("Simulation complete: %d frames", len(snapshots))

    output = pathlib.Path(args.output)
    log.info("Saving GIF to %s", output)
    animate(snapshots, path=str(output), fps=args.fps)
    log.info("Done.")


def _run_simulation(graph, func, treatments_per_step: int, max_steps: int, rng: np.random.Generator) -> list:
    snapshots = [copy.deepcopy(graph)]
    for _ in range(max_steps):
        burning = [n for n in graph.nodes if graph.nodes[n][STATE] == NodeState.BURNING]
        if not burning:
            break
        precompute_fire_map(graph)
        precompute_burnable_fire_map(graph)
        graph.graph[TREATMENTS_REMAINING] = treatments_per_step
        _apply_treatments(graph, func, treatments_per_step)
        spread_step(graph, rng)
        snapshots.append(copy.deepcopy(graph))
    return snapshots


def _apply_treatments(graph, func, budget: int) -> None:
    candidates = [n for n in graph.nodes if graph.nodes[n][STATE] == NodeState.UNBURNED]
    candidates.sort(key=lambda n: _safe_score(func, graph, n), reverse=True)
    for node in candidates[:budget]:
        graph.nodes[node][STATE] = NodeState.TREATED
        graph.graph[TREATMENTS_REMAINING] -= 1


def _safe_score(func, graph, node) -> float:
    import math as _math

    score = func(graph, node)
    return score if _math.isfinite(score) else float("-inf")


def _load_strategy(args: argparse.Namespace) -> tuple:
    if args.hof:
        path = pathlib.Path(args.hof)
        with open(path, "rb") as f:
            func = dill.load(f)
        return func, path.stem
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
    add_landscape_args(parser)
    parser.add_argument("--output", type=str, default="simulation.gif")
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
