"""
Batch-evaluate GP candidates from a results directory across N landscapes and M runs per landscape.

Loads the final population (final_population.expr) and HOF individuals (hof_*.expr) from a results directory,
deduplicates by expression string, evaluates each candidate across --landscapes independent landscapes with --runs
simulations per landscape, and prints a ranked table of mean ± std burned nodes.

All candidates are evaluated on the same set of landscapes for a fair comparison. The evaluation landscapes are
independent of the training landscape by default (seed=None). Pass --seed for a reproducible comparison.

Natural extinction filtering
----------------------------
Some ignition points produce fires that die out regardless of strategy, due to low local fuel connectivity. Including
these runs would conflate natural extinction with genuine strategy performance and drag the mean toward zero. To avoid
this, each ignition point is validated upfront with a single no-treatment probe: if the fire burns fewer than
--min-burned nodes without any intervention, the ignition is discarded and a new one is sampled (up to --max-ignition-
tries attempts). Only ignition points where fire spreads meaningfully are used for candidate evaluation.

Deduplication
-------------
Candidates are identified by their expression string. If the same expression appears in both the final population and
the HOF, it is evaluated once.

Usage
-----
    python -m scripts.batch_evaluate --results-dir PATH
                                     [--landscapes INT] [--runs INT] [--seed INT] [--output PATH]
                                     [--min-burned INT] [--max-ignition-tries INT]
                                     [--rows INT] [--cols INT]
                                     [--treatments INT] [--max-steps INT] [--intervention-delay INT]
                                     [--wind-speed FLOAT] [--wind-direction FLOAT] [--moisture FLOAT]

    --landscapes INT          Number of independent landscapes to evaluate on (default 5).
    --runs INT                Simulations per landscape (default 5).
    --min-burned INT          Minimum burned nodes in a no-treatment probe for the ignition to be
                              considered valid (default 50). Ignitions below this threshold are
                              replaced with a fresh sample.
    --max-ignition-tries INT  Maximum attempts to find a valid ignition per landscape (default 20).
                              If no valid ignition is found, the best available is used with a warning.
    --output PATH             Save the ranked results table as a CSV file. Each row contains rank, mean,
                              std, and the full (untruncated) expression string. Use this to identify
                              candidates for further analysis scripts via --expr.

Examples
--------
    python -m scripts.batch_evaluate --results-dir results/2026-05-11_14-25-31
    python -m scripts.batch_evaluate --results-dir results/2026-05-11_14-25-31 --landscapes 10 --runs 20 --seed 42 \
        --output results/2026-05-11_14-25-31/batch.csv
"""

import argparse
import csv
import logging
import pathlib
import sys

import numpy as np

from scripts.cli import (
    add_landscape_args,
    add_multi_landscape_args,
    find_valid_ignition,
    load_expr,
    read_hof_exprs,
    read_population_exprs,
)
from wildfireGP.evaluate import evaluate
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

    # Pre-generate all seeds so every candidate sees the same landscapes and spread sequences.
    land_seeds = [int(root_rng.integers(2**31)) if args.seed is not None else None for _ in range(args.landscapes)]
    run_seed_matrix = [
        [int(root_rng.integers(2**31)) if args.seed is not None else None for _ in range(args.runs)]
        for _ in range(args.landscapes)
    ]

    # Validate ignitions upfront — shared across all candidates for a fair comparison.
    ignitions = []
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
        ignitions.append(ignition)

    candidates = _load_candidates(args.results_dir)
    log.info(
        "Evaluating %d unique candidates across %d landscapes x %d runs",
        len(candidates),
        args.landscapes,
        args.runs,
    )

    results = []
    for i, (expr, func) in enumerate(candidates, 1):
        log.info("  %d/%d  %s", i, len(candidates), expr[:80])
        burned_counts = []
        for land_seed, ignition, run_seeds in zip(land_seeds, ignitions, run_seed_matrix):
            graph = create_grid(args.rows, args.cols, seed=land_seed)
            set_wind(graph, speed=args.wind_speed, direction=args.wind_direction)
            set_fuel_moisture(graph, moisture=args.moisture)
            for run_seed in run_seeds:
                b, _ = evaluate(
                    func,
                    graph,
                    [ignition],
                    args.treatments,
                    args.max_steps,
                    np.random.default_rng(run_seed),
                    args.intervention_delay,
                )
                burned_counts.append(b)
        results.append((expr, float(np.mean(burned_counts)), float(np.std(burned_counts))))

    results.sort(key=lambda x: x[1])
    _print_table(results)
    if args.output:
        _save_csv(pathlib.Path(args.output), results)
        log.info("Saved CSV to %s", args.output)


def _load_candidates(results_dir: pathlib.Path) -> list[tuple[str, object]]:
    """
    Load and deduplicate candidates from final_population.expr and hof_*.expr files.

    Each expression string is compiled once against the current PRIMITIVE_SET via load_expr. Duplicate expressions
    across the population and HOF are evaluated once.
    """
    candidates: dict[str, object] = {}

    pop_exprs = read_population_exprs(results_dir)
    for expr in pop_exprs:
        if expr not in candidates:
            candidates[expr] = load_expr(expr)
    if pop_exprs:
        log.info("Loaded %d individuals from final_population.expr", len(pop_exprs))

    for _, expr in read_hof_exprs(results_dir):
        if expr not in candidates:
            candidates[expr] = load_expr(expr)

    if not candidates:
        raise SystemExit(
            f"No candidates found in {results_dir}. Expected final_population.expr and/or hof_*.expr files."
        )

    return list(candidates.items())


def _save_csv(path: pathlib.Path, results: list[tuple[str, float, float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "mean", "std", "expr"])
        for rank, (expr, mean, std) in enumerate(results, 1):
            writer.writerow([rank, f"{mean:.1f}", f"{std:.1f}", expr])


def _print_table(results: list[tuple[str, float, float]]) -> None:
    print(f"\n{'rank':>4}  {'mean':>8}  {'std':>8}  expr")
    print("-" * 120)
    for rank, (expr, mean, std) in enumerate(results, 1):
        print(f"{rank:>4}  {mean:>8.1f}  {std:>8.1f}  {expr}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate GP candidates from a results directory.")
    parser.add_argument(
        "--results-dir", type=pathlib.Path, required=True, help="Path to a run_gp.py results directory."
    )
    parser.add_argument("--runs", type=int, default=5, help="Simulations per landscape (default 5).")
    parser.add_argument("--output", type=str, default=None, help="Save ranked results as a CSV file.")
    add_landscape_args(parser)
    add_multi_landscape_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
