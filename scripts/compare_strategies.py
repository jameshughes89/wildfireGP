"""
Compare GP-evolved strategies against baseline heuristics.

Each strategy is evaluated over --runs independent simulations on the same landscape and ignition point. Mean and
standard deviation of total_burned and peak_burning are reported in a ranked table (ranked by mean total_burned,
ascending).

Usage
-----
    python -m scripts.compare_strategies [--results-dir PATH] [--seed INT] [--rows INT] [--cols INT]
                                         [--treatments INT] [--max-steps INT] [--intervention-delay INT]
                                         [--wind-speed FLOAT] [--wind-direction FLOAT] [--moisture FLOAT]
                                         [--runs INT] [--hof PATH [PATH ...]] [--expr STRING]

    --hof accepts one or more paths to .dill files produced by run_gp.py. If omitted only the builtin baselines are
    compared. If --results-dir is given without --expr, all hof_*.dill files found directly inside that directory are
    loaded automatically.

    --expr STRING  Expression string of a specific GP candidate to load from --results-dir (required alongside --expr).
                   Copy the expr value from the batch_evaluate.py CSV output. --expr is mutually exclusive with --hof.

Examples
--------
    # baselines only
    python -m scripts.compare_strategies

    # load every HOF individual from a specific run directory
    python -m scripts.compare_strategies --results-dir results/2026-05-06_14-32-00

    # load specific dill files
    python -m scripts.compare_strategies --hof results/2026-05-06_14-32-00/hof_0.dill

    # load a candidate by expression string from batch_evaluate CSV
    python -m scripts.compare_strategies --results-dir results/2026-05-06_14-32-00 --expr "min(fuel_level, ...)"
"""

import argparse
import logging
import pathlib
import sys

import dill
import numpy as np

from scripts.cli import add_landscape_args, load_candidate_by_expr
from wildfireGP.evaluate import DEFAULT_INTERVENTION_DELAY, evaluate
from wildfireGP.network import (
    create_grid,
    select_ignition_node,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.strategies import ALL_STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_COL_WIDTH = 24
_NUM_WIDTH = 10


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
    log.info("Evaluating %d strategies + no_treatment baseline over %d runs each", len(strategies), args.runs)

    results = []
    log.info("  no_treatment ...")
    burned, peak = _run_strategy(
        lambda g, n: 0.0, graph, [ignition], 0, args.max_steps, args.runs, args.seed, args.intervention_delay
    )
    results.append(("no_treatment", burned.mean(), burned.std(), peak.mean(), peak.std()))

    for name, func in strategies:
        log.info("  %s ...", name)
        burned, peak = _run_strategy(
            func, graph, [ignition], args.treatments, args.max_steps, args.runs, args.seed, args.intervention_delay
        )
        results.append((name, burned.mean(), burned.std(), peak.mean(), peak.std()))

    results.sort(key=lambda r: r[1])
    _print_table(results)


def _run_strategy(
    func,
    graph,
    ignition_nodes: list[tuple],
    treatments_per_step: int,
    max_steps: int,
    runs: int,
    base_seed: int | None,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> tuple[np.ndarray, np.ndarray]:
    burned = np.empty(runs, dtype=float)
    peak = np.empty(runs, dtype=float)
    for i in range(runs):
        seed = None if base_seed is None else base_seed + i
        rng = np.random.default_rng(seed)
        b, p = evaluate(func, graph, ignition_nodes, treatments_per_step, max_steps, rng, intervention_delay)
        burned[i] = b
        peak[i] = p
    return burned, peak


def _load_strategies(args: argparse.Namespace) -> list[tuple[str, object]]:
    strategies = [(f.__name__, f) for f in ALL_STRATEGIES]

    if getattr(args, "expr", None):
        if not args.results_dir:
            raise SystemExit("--results-dir is required when using --expr.")
        func = load_candidate_by_expr(args.results_dir, args.expr)
        strategies.append((args.expr[:60], func))
        return strategies

    dill_paths: list[pathlib.Path] = list(args.hof) if args.hof else []
    if args.results_dir:
        dill_paths += sorted(args.results_dir.glob("hof_*.dill"))

    seen: set[pathlib.Path] = set()
    for path in dill_paths:
        path = pathlib.Path(path).resolve()
        if path in seen:
            continue
        seen.add(path)
        with open(path, "rb") as f:
            func = dill.load(f)
        strategies.append((path.stem, func))

    return strategies


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GP and baseline strategies.")
    add_landscape_args(parser)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--results-dir", type=pathlib.Path, default=None)
    gp_source = parser.add_mutually_exclusive_group()
    gp_source.add_argument("--hof", type=pathlib.Path, nargs="+", default=None)
    gp_source.add_argument("--expr", type=str, default=None, help="Load a candidate by expression string.")
    return parser.parse_args(argv)


def _print_table(results: list[tuple]) -> None:
    w = _COL_WIDTH
    n = _NUM_WIDTH
    header = f"{'strategy':<{w}} {'burned_mean':>{n}} {'burned_std':>{n}} {'peak_mean':>{n}} {'peak_std':>{n}}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for name, bm, bs, pm, ps in results:
        print(f"{name:<{w}} {bm:>{n}.2f} {bs:>{n}.2f} {pm:>{n}.2f} {ps:>{n}.2f}")
    print(separator)


if __name__ == "__main__":
    main(sys.argv[1:])
