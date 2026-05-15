"""
Batch-evaluate GP candidates from a results directory across N landscapes and M runs per landscape.

Loads the final population (final_population.dill) and HOF individuals (hof_*.dill + hof_*.expr) from a results
directory, deduplicates by expression string, evaluates each candidate across --landscapes independent landscapes with
--runs simulations per landscape, and prints a ranked table of mean ± std burned nodes.

All candidates are evaluated on the same set of landscapes for a fair comparison. The evaluation landscapes are
independent of the training landscape by default (seed=None). Pass --seed for a reproducible comparison.

Deduplication
-------------
Candidates are identified by their expression string. If the same expression appears in both the final population and
the HOF, it is evaluated once. HOF individuals from runs predating the final_population.dill save (i.e. runs before this
feature was added) can still be evaluated via the hof_*.dill + hof_*.expr files alone.

Usage
-----
    python -m scripts.batch_evaluate --results-dir PATH
                                     [--landscapes INT] [--runs INT] [--seed INT] [--output PATH]
                                     [--rows INT] [--cols INT]
                                     [--treatments INT] [--max-steps INT] [--intervention-delay INT]
                                     [--wind-speed FLOAT] [--wind-direction FLOAT] [--moisture FLOAT]

    --landscapes INT  Number of independent landscapes to evaluate on (default 5).
    --runs INT        Simulations per landscape (default 10).
    --output PATH     Save the ranked results table as a CSV file. Each row contains rank, mean,
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

import dill
import numpy as np

from scripts.cli import add_landscape_args
from wildfireGP.evaluate import evaluate
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

    root_rng = np.random.default_rng(args.seed)

    # Pre-generate all seeds so every candidate sees the same landscapes and spread sequences.
    land_seeds = [int(root_rng.integers(2**31)) if args.seed is not None else None for _ in range(args.landscapes)]
    run_seed_matrix = [
        [int(root_rng.integers(2**31)) if args.seed is not None else None for _ in range(args.runs)]
        for _ in range(args.landscapes)
    ]

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
        for land_idx, land_seed in enumerate(land_seeds):
            graph = create_grid(args.rows, args.cols, seed=land_seed)
            set_wind(graph, speed=args.wind_speed, direction=args.wind_direction)
            set_fuel_moisture(graph, moisture=args.moisture)
            ignition = select_ignition_node(graph, np.random.default_rng(land_seed))
            for run_seed in run_seed_matrix[land_idx]:
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
    Load and deduplicate candidates from final_population.dill and hof_*.dill files.

    final_population.dill takes precedence for deduplication: if the same expression appears in
    both the population and the HOF, the population's compiled callable is used. HOF-only entries
    (expressions not present in the population) are appended after the population.
    """
    candidates: dict[str, object] = {}

    pop_path = results_dir / "final_population.dill"
    if pop_path.exists():
        with open(pop_path, "rb") as f:
            population = dill.load(f)
        for expr, func in population:
            candidates[expr] = func
        log.info("Loaded %d individuals from final_population.dill", len(population))

    for hof_path in sorted(results_dir.glob("hof_*.dill")):
        expr_path = hof_path.with_suffix(".expr")
        if not expr_path.exists():
            continue
        expr = expr_path.read_text().strip()
        if expr not in candidates:
            with open(hof_path, "rb") as f:
                candidates[expr] = dill.load(f)

    if not candidates:
        raise SystemExit(
            f"No candidates found in {results_dir}. "
            "Expected final_population.dill and/or hof_*.dill + hof_*.expr files."
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
    parser.add_argument("--landscapes", type=int, default=5, help="Number of independent landscapes (default 5).")
    parser.add_argument("--runs", type=int, default=10, help="Simulations per landscape (default 10).")
    parser.add_argument("--output", type=str, default=None, help="Save ranked results as a CSV file.")
    add_landscape_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
