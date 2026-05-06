"""
Run the wildfire GP and save results to a timestamped directory under results/.

Usage
-----
    python -m scripts.run_gp [--results-dir PATH] [--seed INT] [--rows INT] [--cols INT]
                             [--treatments INT] [--max-steps INT]
                             [--pop INT] [--gens INT] [--hof INT]

Outputs (one directory per run)
--------------------------------
    results/<timestamp>/
        config.json     GPConfig + scenario parameters (human-readable, for reproducibility)
        stats.json      per-generation fitness and size statistics from the DEAP logbook
        population.pkl  full DEAP population and logbook (pickle)
        hof_0.dill      best individual compiled to a callable via dill
        hof_1.dill      second-best, etc. (up to --hof individuals)
        hof_0.expr      tree expression string for hof_0 (human-readable)
        hof_1.expr      ...
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
from deap import gp, tools

from wildfireGP.gp import GPConfig, build_toolbox, run
from wildfireGP.language import PRIMITIVE_SET
from wildfireGP.network import create_grid, select_ignition_node, set_fuel_moisture, set_wind

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = pathlib.Path("results")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    log.info("Building landscape (%dx%d, seed=%s)", args.rows, args.cols, args.seed)
    graph = create_grid(args.rows, args.cols, seed=args.seed)
    set_wind(graph, speed=20.0, direction=0.0)
    set_fuel_moisture(graph, moisture=0.2)

    ignition = select_ignition_node(graph, rng)
    log.info("Ignition node: %s", ignition)

    config = GPConfig(
        population_size=args.pop,
        generations=args.gens,
    )

    scenario = {
        "rows": args.rows,
        "cols": args.cols,
        "seed": args.seed,
        "ignition_node": list(ignition),
        "treatments_per_step": args.treatments,
        "max_steps": args.max_steps,
        "wind_speed": 20.0,
        "wind_direction": 0.0,
        "fuel_moisture": 0.2,
    }

    log.info("Starting GP: %d individuals, %d generations", config.population_size, config.generations)
    population, logbook = run(config, graph, [ignition], args.treatments, args.max_steps, rng)

    out_dir = _make_output_dir(args.results_dir)
    log.info("Saving results to %s", out_dir)

    _save_config(out_dir, config, scenario)
    _save_stats(out_dir, logbook)
    _save_population(out_dir, population, logbook)
    _save_hof(out_dir, population, args.hof, build_toolbox(graph, [ignition], args.treatments, args.max_steps, rng, config))

    log.info("Done. Results in %s", out_dir)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wildfire GP and save results.")
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--treatments", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=50)
    parser.add_argument("--hof", type=int, default=5, help="Number of best individuals to save.")
    return parser.parse_args(argv)


def _make_output_dir(base: pathlib.Path) -> pathlib.Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base / timestamp
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


def _save_hof(out_dir: pathlib.Path, population: list, n: int, toolbox) -> None:
    hof = tools.selBest(population, min(n, len(population)))
    for i, ind in enumerate(hof):
        compiled = toolbox.compile(ind)
        with open(out_dir / f"hof_{i}.dill", "wb") as f:
            dill.dump(compiled, f)
        (out_dir / f"hof_{i}.expr").write_text(str(ind))
        log.info("hof_%d fitness=%s expr=%s", i, ind.fitness.values, str(ind))


if __name__ == "__main__":
    main(sys.argv[1:])
