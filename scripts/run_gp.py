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
        final_population.expr   final generation as one expression string per line; recompile against
                                the current PRIMITIVE_SET via scripts.cli.load_expr
        hof_0.expr              best individual's expression string (recompile via load_expr)
        hof_1.expr              second-best, etc. (up to --hof individuals)
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

import numpy as np
from deap import tools

from scripts.cli import (
    add_landscape_args,
    add_wind_per_landscape_args,
    landscape_kwargs,
    resolve_landscape_count,
    resolve_wind_for_landscape,
)
from wildfireGP.gp import GPConfig, run
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

    landscapes_count = resolve_landscape_count(args)

    scenarios = []
    land_ignitions: list[list[tuple]] = []
    wind_per_landscape: list[dict[str, float]] = []
    for i in range(landscapes_count):
        land_seed = None if args.seed is None else args.seed + i
        wind_speed, wind_direction = resolve_wind_for_landscape(args, i, rng)
        log.info(
            "Building landscape %d/%d (%dx%d, seed=%s, wind=%.1f km/h @ %.1f°)",
            i + 1,
            landscapes_count,
            args.rows,
            args.cols,
            land_seed,
            wind_speed,
            wind_direction,
        )
        graph = create_grid(**landscape_kwargs(args), seed=land_seed)
        set_wind(graph, speed=wind_speed, direction=wind_direction)
        set_fuel_moisture(graph, moisture=args.moisture)
        ignition_nodes = select_ignition_cluster(graph, rng, size=args.ignition_cluster_size)
        log.info("  ignition cluster (%d nodes): %s", len(ignition_nodes), ignition_nodes)
        scenarios.append((graph, ignition_nodes))
        land_ignitions.append(ignition_nodes)
        wind_per_landscape.append({"speed": wind_speed, "direction": wind_direction})

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
        "landscapes": landscapes_count,
        "ignition_nodes_per_landscape": [[list(n) for n in cluster] for cluster in land_ignitions],
        "treatments_per_step": args.treatments,
        "max_steps": args.max_steps,
        "intervention_delay": args.intervention_delay,
        "min_treatment_distance": args.min_treatment_distance,
        "terrain_smoothing": args.terrain_smoothing,
        "fuel_smoothing": args.fuel_smoothing,
        "water_fraction": args.water_fraction,
        "rock_fraction": args.rock_fraction,
        "wind_per_landscape": wind_per_landscape,
        "fuel_moisture": args.moisture,
    }

    log.info(
        "Starting GP: %d individuals, %d generations, %d landscape(s)",
        config.population_size,
        config.generations,
        landscapes_count,
    )
    population, logbook, hof = run(
        config,
        scenarios,
        args.treatments,
        args.max_steps,
        rng,
        args.intervention_delay,
        hof_size=args.hof,
        min_treatment_distance=args.min_treatment_distance,
    )

    out_dir = _make_output_dir(args.results_dir, args.run_name)
    log.info("Saving results to %s", out_dir)

    _save_config(out_dir, config, scenario)
    _save_stats(out_dir, logbook)
    _save_population(out_dir, population, logbook)
    _save_final_population_expr(out_dir, population)
    _save_hof(out_dir, hof)

    log.info("Done. Results in %s", out_dir)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wildfire GP and save results.")
    add_landscape_args(parser)
    add_wind_per_landscape_args(parser)
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ignition-cluster-size", type=int, default=3, help="Number of nodes to ignite at t=0.")
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


def _save_final_population_expr(out_dir: pathlib.Path, population: list) -> None:
    lines = [str(ind) for ind in population]
    (out_dir / "final_population.expr").write_text("\n".join(lines) + "\n")


def _save_hof(out_dir: pathlib.Path, hof: tools.HallOfFame) -> None:
    for i, ind in enumerate(hof):
        (out_dir / f"hof_{i}.expr").write_text(str(ind))
        log.info("hof_%d fitness=%s expr=%s", i, ind.fitness.values, str(ind))


if __name__ == "__main__":
    main(sys.argv[1:])
