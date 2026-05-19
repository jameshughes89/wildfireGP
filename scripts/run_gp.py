"""
Run the wildfire GP and save results to a timestamped directory under results/.

Usage
-----
    python -m scripts.run_gp [--results-dir PATH] [--run-name NAME]
                             [--seed INT] [--rows INT] [--cols INT]
                             [--treatments INT] [--max-steps INT] [--intervention-delay INT]
                             [--wind-speed FLOAT] [--wind-direction FLOAT] [--moisture FLOAT]
                             [--ignition-cluster-size INT]
                             [--pop INT] [--gens INT] [--hof INT]
                             [--crossover-prob FLOAT] [--mutation-prob FLOAT]
                             [--tournament-size INT] [--max-tree-height INT] [--max-tree-nodes INT]
                             [--init-min-height INT] [--init-max-height INT]
                             [--mutation-min-height INT] [--mutation-max-height INT]

--run-name overrides the auto-generated timestamp subdirectory name, which is useful for naming
factorial experiment runs (e.g. "A_small_tourn2", "B_large_tourn3").

Outputs (one directory per run)
--------------------------------
    results/<timestamp>/          (or results/<run-name>/ when --run-name is given)
        config.json             GPConfig + scenario parameters (human-readable, for reproducibility)
        stats.json              per-generation fitness and size statistics from the DEAP logbook
        population.pkl          full DEAP population and logbook (pickle); requires DEAP types
                                registered on load — use for GP resume or custom post-processing
        final_population.dill   final generation as [(expr_string, callable)] pairs (dill);
                                self-contained, no DEAP dependency — use with batch_evaluate.py
        hof_0.dill              best individual compiled to a callable via dill; self-contained
        hof_1.dill              second-best, etc. (up to --hof individuals)
        hof_0.expr              tree expression string for hof_0 (human-readable)
        hof_1.expr              ...

population.pkl vs final_population.dill
----------------------------------------
population.pkl stores raw DEAP PrimitiveTree objects and requires DEAP creator types to be
registered before unpickling. final_population.dill stores pre-compiled (expr, callable) pairs
via dill, which is self-contained and can be loaded anywhere without DEAP. Use population.pkl
if you need access to the tree structure; use final_population.dill for evaluation and comparison
via batch_evaluate.py.
"""

import argparse
import dataclasses
import datetime
import json
import logging
import pathlib
import pickle
import random
import sys

import dill
import numpy as np
from deap import tools

from scripts.cli import add_landscape_args
from wildfireGP.gp import GPConfig, compile_individual, run
from wildfireGP.network import (
    create_grid,
    select_ignition_cluster,
    set_fuel_moisture,
    set_wind,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = pathlib.Path("results")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    scenarios = []
    land_ignitions: list[list[tuple]] = []
    for i in range(args.landscapes):
        land_seed = None if args.seed is None else args.seed + i
        log.info("Building landscape %d/%d (%dx%d, seed=%s)", i + 1, args.landscapes, args.rows, args.cols, land_seed)
        graph = create_grid(args.rows, args.cols, seed=land_seed)
        set_wind(graph, speed=args.wind_speed, direction=args.wind_direction)
        set_fuel_moisture(graph, moisture=args.moisture)
        ignition_nodes = select_ignition_cluster(graph, rng, size=args.ignition_cluster_size)
        log.info("  ignition cluster (%d nodes): %s", len(ignition_nodes), ignition_nodes)
        scenarios.append((graph, ignition_nodes))
        land_ignitions.append(ignition_nodes)

    config = GPConfig(
        population_size=args.pop,
        generations=args.gens,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        tournament_size=args.tournament_size,
        max_tree_height=args.max_tree_height,
        max_tree_nodes=args.max_tree_nodes,
        init_min_height=args.init_min_height,
        init_max_height=args.init_max_height,
        mutation_min_height=args.mutation_min_height,
        mutation_max_height=args.mutation_max_height,
    )

    scenario = {
        "rows": args.rows,
        "cols": args.cols,
        "seed": args.seed,
        "landscapes": args.landscapes,
        "ignition_nodes_per_landscape": [[list(n) for n in cluster] for cluster in land_ignitions],
        "treatments_per_step": args.treatments,
        "max_steps": args.max_steps,
        "intervention_delay": args.intervention_delay,
        "wind_speed": args.wind_speed,
        "wind_direction": args.wind_direction,
        "fuel_moisture": args.moisture,
    }

    log.info(
        "Starting GP: %d individuals, %d generations, %d landscape(s)",
        config.population_size,
        config.generations,
        args.landscapes,
    )
    population, logbook, hof = run(
        config, scenarios, args.treatments, args.max_steps, rng, args.intervention_delay, hof_size=args.hof
    )

    out_dir = _make_output_dir(args.results_dir, args.run_name)
    log.info("Saving results to %s", out_dir)

    _save_config(out_dir, config, scenario)
    _save_stats(out_dir, logbook)
    _save_population(out_dir, population, logbook)
    _save_final_population_dill(out_dir, population)
    _save_hof(out_dir, hof)

    log.info("Done. Results in %s", out_dir)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wildfire GP and save results.")
    add_landscape_args(parser)
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ignition-cluster-size", type=int, default=3, help="Number of nodes to ignite at t=0.")
    parser.add_argument(
        "--landscapes",
        type=int,
        default=3,
        help="Number of landscapes to evaluate each candidate on per generation. Fitness is mean total_burned "
        "across landscapes. Default 3.",
    )
    parser.add_argument("--pop", type=int, default=GPConfig.population_size)
    parser.add_argument("--gens", type=int, default=GPConfig.generations)
    parser.add_argument("--hof", type=int, default=5, help="Number of best individuals to save.")
    parser.add_argument("--tournament-size", type=int, default=GPConfig.tournament_size)
    parser.add_argument("--init-min-height", type=int, default=GPConfig.init_min_height)
    parser.add_argument("--init-max-height", type=int, default=GPConfig.init_max_height)
    parser.add_argument("--mutation-min-height", type=int, default=GPConfig.mutation_min_height)
    parser.add_argument("--mutation-max-height", type=int, default=GPConfig.mutation_max_height)
    parser.add_argument("--max-tree-height", type=int, default=GPConfig.max_tree_height)
    parser.add_argument("--max-tree-nodes", type=int, default=GPConfig.max_tree_nodes)
    parser.add_argument("--crossover-prob", type=float, default=GPConfig.crossover_prob)
    parser.add_argument("--mutation-prob", type=float, default=GPConfig.mutation_prob)
    parser.add_argument("--run-name", type=str, default=None, help="Override the timestamp subdirectory name.")
    return parser.parse_args(argv)


def _make_output_dir(base: pathlib.Path, run_name: str | None = None) -> pathlib.Path:
    name = run_name if run_name else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_config(out_dir: pathlib.Path, config: GPConfig, scenario: dict) -> None:
    payload = {"gp_config": dataclasses.asdict(config), "scenario": scenario}
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2))


def _save_stats(out_dir: pathlib.Path, logbook: tools.Logbook) -> None:
    chapter_data: dict[int, dict] = {}
    for section, chapter in logbook.chapters.items():
        for record in chapter:
            gen = record["gen"]
            if gen not in chapter_data:
                chapter_data[gen] = {"gen": gen}
            for stat, val in record.items():
                if stat == "gen":
                    continue
                chapter_data[gen][f"{section}_{stat}"] = val.tolist() if hasattr(val, "tolist") else val

    rows = [chapter_data[g] for g in sorted(chapter_data)]
    (out_dir / "stats.json").write_text(json.dumps(rows, indent=2))


def _save_population(out_dir: pathlib.Path, population: list, logbook: tools.Logbook) -> None:
    payload = {"population": population, "logbook": logbook}
    with open(out_dir / "population.pkl", "wb") as f:
        pickle.dump(payload, f)


def _save_final_population_dill(out_dir: pathlib.Path, population: list) -> None:
    entries = [(str(ind), compile_individual(ind)) for ind in population]
    with open(out_dir / "final_population.dill", "wb") as f:
        dill.dump(entries, f)


def _save_hof(out_dir: pathlib.Path, hof: tools.HallOfFame) -> None:
    for i, ind in enumerate(hof):
        compiled = compile_individual(ind)
        with open(out_dir / f"hof_{i}.dill", "wb") as f:
            dill.dump(compiled, f)
        (out_dir / f"hof_{i}.expr").write_text(str(ind))
        log.info("hof_%d fitness=%s expr=%s", i, ind.fitness.values, str(ind))


if __name__ == "__main__":
    main(sys.argv[1:])
