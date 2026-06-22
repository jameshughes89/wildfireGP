"""
Compare GP-evolved strategies against baseline heuristics.

Each strategy is evaluated across --landscapes independent landscapes with --runs simulations per landscape. Mean and
standard deviation of total_burned and peak_burning are reported in a ranked table (ranked by mean total_burned,
ascending). All strategies share the same set of landscapes for a fair comparison.

Usage
-----
    python -m scripts.compare_strategies [--results-dir PATH] [--seed INT] [--rows INT] [--cols INT]
                                         [--treatments INT] [--max-steps INT] [--intervention-delay INT]
                                         [--wind-speed FLOAT] [--wind-direction FLOAT] [--moisture FLOAT]
                                         [--landscapes INT] [--runs INT] [--hof PATH [PATH ...]] [--expr STRING]

    --hof accepts one or more paths to .expr files produced by run_gp.py. If omitted only the builtin baselines are
    compared. If --results-dir is given without --expr, all hof_*.expr files found directly inside that directory are
    loaded automatically.

    --expr STRING  Expression string of a specific GP candidate to load from --results-dir (required alongside --expr).
                   Copy the expr value from the batch_evaluate.py CSV output. --expr is mutually exclusive with --hof.

Examples
--------
    # baselines only
    python -m scripts.compare_strategies

    # load every HOF individual from a specific run directory
    python -m scripts.compare_strategies --results-dir results/2026-05-06_14-32-00

    # load specific expr files
    python -m scripts.compare_strategies --hof results/2026-05-06_14-32-00/hof_0.expr

    # load a candidate by expression string from batch_evaluate CSV
    python -m scripts.compare_strategies --results-dir results/2026-05-06_14-32-00 --expr "min(fuel_level, ...)"
"""

import argparse
import logging
import pathlib
import sys

import numpy as np

from scripts.cli import (
    add_landscape_args,
    add_multi_landscape_args,
    add_wind_per_landscape_args,
    find_valid_ignition,
    landscape_kwargs,
    load_candidate_by_expr,
    load_expr,
    resolve_landscape_count,
    resolve_wind_for_landscape,
)
from wildfireGP.evaluate import DEFAULT_INTERVENTION_DELAY, evaluate
from wildfireGP.network import (
    create_grid,
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

    root_rng = np.random.default_rng(args.seed)
    landscapes_count = resolve_landscape_count(args)

    land_seeds = [int(root_rng.integers(2**31)) if args.seed is not None else None for _ in range(landscapes_count)]
    wind_per_landscape = [resolve_wind_for_landscape(args, i, root_rng) for i in range(landscapes_count)]
    run_seed_matrix = [
        [int(root_rng.integers(2**31)) if args.seed is not None else None for _ in range(args.runs)]
        for _ in range(landscapes_count)
    ]

    grid_kwargs = landscape_kwargs(args)

    log.info("Validating ignitions across %d landscapes", landscapes_count)
    landscapes = []
    for land_seed, (wind_speed, wind_direction) in zip(land_seeds, wind_per_landscape):
        graph = create_grid(**grid_kwargs, seed=land_seed)
        set_wind(graph, speed=wind_speed, direction=wind_direction)
        set_fuel_moisture(graph, moisture=args.moisture)
        ignition = find_valid_ignition(
            graph,
            np.random.default_rng(land_seed),
            args.max_steps,
            args.intervention_delay,
            args.min_burned,
            args.max_ignition_tries,
        )
        landscapes.append((land_seed, wind_speed, wind_direction, ignition))

    strategies = _load_strategies(args)
    n_extra = 0 if args.no_baselines else 1
    log.info(
        "Evaluating %d strategies%s over %d landscapes x %d runs",
        len(strategies) + n_extra,
        "" if args.no_baselines else " + no_treatment",
        landscapes_count,
        args.runs,
    )

    results = []
    raw_rows: list[tuple[str, int, int, float, float]] = []

    def _record_raw(name: str, burned: np.ndarray, peak: np.ndarray) -> None:
        if args.save_raw is None:
            return
        for i, (b, p) in enumerate(zip(burned, peak)):
            raw_rows.append((name, i // args.runs, i % args.runs, float(b), float(p)))

    if not args.no_baselines:
        log.info("  no_treatment ...")
        burned, peak = _run_strategy(
            lambda g, n: 0.0,
            landscapes,
            0,
            grid_kwargs,
            args.moisture,
            args.max_steps,
            run_seed_matrix,
            args.intervention_delay,
            args.min_treatment_distance,
        )
        results.append(("no_treatment", burned.mean(), burned.std(), peak.mean(), peak.std()))
        _record_raw("no_treatment", burned, peak)

    for name, func in strategies:
        log.info("  %s ...", name)
        burned, peak = _run_strategy(
            func,
            landscapes,
            args.treatments,
            grid_kwargs,
            args.moisture,
            args.max_steps,
            run_seed_matrix,
            args.intervention_delay,
            args.min_treatment_distance,
        )
        results.append((name, burned.mean(), burned.std(), peak.mean(), peak.std()))
        _record_raw(name, burned, peak)

    results.sort(key=lambda r: r[1])
    _print_table(results)

    if args.save_raw is not None:
        _save_raw_csv(args.save_raw, raw_rows)
        log.info("Saved per-run results to %s", args.save_raw)


def _run_strategy(
    func,
    landscapes: list[tuple],
    treatments_per_step: int,
    grid_kwargs: dict,
    moisture: float,
    max_steps: int,
    run_seed_matrix: list[list],
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
    min_treatment_distance: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    burned_all = []
    peak_all = []
    for (land_seed, wind_speed, wind_direction, ignition), run_seeds in zip(landscapes, run_seed_matrix):
        graph = create_grid(**grid_kwargs, seed=land_seed)
        set_wind(graph, speed=wind_speed, direction=wind_direction)
        set_fuel_moisture(graph, moisture=moisture)
        for run_seed in run_seeds:
            b, p = evaluate(
                func,
                graph,
                [ignition],
                treatments_per_step,
                max_steps,
                np.random.default_rng(run_seed),
                intervention_delay,
                min_treatment_distance,
            )
            burned_all.append(b)
            peak_all.append(p)
    return np.array(burned_all, dtype=float), np.array(peak_all, dtype=float)


def _load_strategies(args: argparse.Namespace) -> list[tuple[str, object]]:
    strategies = [] if getattr(args, "no_baselines", False) else [(f.__name__, f) for f in ALL_STRATEGIES]

    if getattr(args, "expr", None):
        if not args.results_dir:
            raise SystemExit("--results-dir is required when using --expr.")
        func = load_candidate_by_expr(args.results_dir, args.expr)
        strategies.append((args.expr[:60], func))
        return strategies

    expr_paths: list[pathlib.Path] = list(args.hof) if args.hof else []
    if args.results_dir:
        expr_paths += sorted(args.results_dir.glob("hof_*.expr"))

    seen: set[pathlib.Path] = set()
    for path in expr_paths:
        path = pathlib.Path(path).resolve()
        if path in seen:
            continue
        seen.add(path)
        expr = path.read_text().strip()
        strategies.append((path.stem, load_expr(expr)))

    return strategies


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GP and baseline strategies.")
    add_landscape_args(parser)
    add_wind_per_landscape_args(parser)
    add_multi_landscape_args(parser)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--results-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip ALL_STRATEGIES and no_treatment; only evaluate --hof / --expr strategies. "
        "Use this to add new GP candidates to an existing comparison without recomputing baselines.",
    )
    parser.add_argument(
        "--save-raw",
        type=pathlib.Path,
        default=None,
        help="If set, write per-(strategy, landscape, run) results to this CSV file. "
        "Columns: strategy, landscape_idx, run_idx, burned, peak. Useful for "
        "distribution comparison (KS test, bootstrap CIs, box plots).",
    )
    gp_source = parser.add_mutually_exclusive_group()
    gp_source.add_argument("--hof", type=pathlib.Path, nargs="+", default=None)
    gp_source.add_argument("--expr", type=str, default=None, help="Load a candidate by expression string.")
    return parser.parse_args(argv)


def _save_raw_csv(path: pathlib.Path, rows: list[tuple[str, int, int, float, float]]) -> None:
    """Save per-(strategy, landscape, run) results as a CSV for distribution analysis.

    Columns: strategy, landscape_idx, run_idx, burned, peak. Creates parent directories
    if needed.
    """
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = _csv.writer(f)
        wr.writerow(["strategy", "landscape_idx", "run_idx", "burned", "peak"])
        wr.writerows(rows)


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
