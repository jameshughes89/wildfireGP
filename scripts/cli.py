"""
Shared CLI utilities for wildfireGP scripts.

add_landscape_args() registers the simulation-scenario arguments common to run_gp.py, compare_strategies.py, and
animate_simulation.py. Centralising them here means defaults and help strings are defined once and all scripts stay in
sync automatically.

add_multi_landscape_args() adds --landscapes, --min-burned, and --max-ignition-tries on top of add_landscape_args().
Use this for evaluation scripts that average results across N independent landscapes.

find_valid_ignition() probes candidate ignition points on a landscape and returns one where fire spreads meaningfully
without treatment, filtering out locations where the fire would naturally extinguish regardless of strategy.

load_candidate_by_expr() loads a compiled GP callable from a results directory by its expression string. Use this to
load a specific candidate identified from batch_evaluate.py output (--output CSV) rather than loading HOF .dill files
by index.
"""

import argparse
import logging
import pathlib

import dill
import numpy as np

from wildfireGP.evaluate import DEFAULT_INTERVENTION_DELAY, evaluate
from wildfireGP.network import select_ignition_node

log = logging.getLogger(__name__)


def load_candidate_by_expr(results_dir: pathlib.Path, expr: str) -> object:
    """
    Load a compiled GP candidate from results_dir by its expression string.

    Searches final_population.dill first, then falls back to hof_*.expr files. Raises SystemExit if the expression
    is not found in either source.
    """
    pop_path = results_dir / "final_population.dill"
    if pop_path.exists():
        with open(pop_path, "rb") as f:
            population = dill.load(f)
        for pop_expr, func in population:
            if pop_expr == expr:
                return func

    for hof_expr_path in sorted(results_dir.glob("hof_*.expr")):
        if hof_expr_path.read_text().strip() == expr:
            hof_dill_path = hof_expr_path.with_suffix(".dill")
            if hof_dill_path.exists():
                with open(hof_dill_path, "rb") as f:
                    return dill.load(f)

    raise SystemExit(f"Expression not found in {results_dir}:\n  {expr[:120]}")


def add_landscape_args(parser: argparse.ArgumentParser) -> None:
    """
    Register the shared landscape and simulation arguments on parser.

    Call this from each script's _parse_args() before adding script-specific arguments.

    Added arguments
    ---------------
    --seed INT                Random seed. Default None draws from OS entropy (non-reproducible).
                              Pass an integer for fully reproducible runs.
    --rows INT                Grid rows. Default 50.
    --cols INT                Grid columns. Default 50.
    --treatments INT          Aggregate treatment budget per timestep across all resources. At 100m/cell
                              this is not individual crew actions but the combined effect of aerial drops,
                              dozer lines, and hand crews. Default 3 reflects a modest initial attack force.
    --max-steps INT           Maximum simulation timesteps before forced termination. Default 100.
    --intervention-delay INT  Steps before any treatment is applied. Models detection, dispatch, and travel
                              time. Default 3 corresponds to roughly 90 minutes to 2 hours at 100m/cell,
                              consistent with initial attack response targets for remote terrain.
    --wind-speed FLOAT        Wind speed in km/h. Default 20.0.
    --wind-direction FLOAT    Wind direction in degrees clockwise from north (0 = north, 90 = east). Default 0.0.
    --moisture FLOAT          Fuel moisture content as a fraction in [0, 1]. Default 0.2.
    """
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None = non-reproducible).")
    parser.add_argument("--rows", type=int, default=50, help="Landscape grid rows.")
    parser.add_argument("--cols", type=int, default=50, help="Landscape grid columns.")
    parser.add_argument("--treatments", type=int, default=3, help="Aggregate treatment budget per timestep.")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum simulation timesteps.")
    parser.add_argument(
        "--intervention-delay",
        type=int,
        default=DEFAULT_INTERVENTION_DELAY,
        help="Steps before treatments begin (models response time).",
    )
    parser.add_argument("--wind-speed", type=float, default=20.0, help="Wind speed in km/h.")
    parser.add_argument(
        "--wind-direction", type=float, default=0.0, help="Wind direction in degrees (0=north, 90=east)."
    )
    parser.add_argument("--moisture", type=float, default=0.2, help="Fuel moisture fraction [0, 1].")


def add_multi_landscape_args(parser: argparse.ArgumentParser) -> None:
    """
    Register multi-landscape evaluation arguments on parser.

    Call this after add_landscape_args() in evaluation scripts that average over N landscapes.

    Added arguments
    ---------------
    --landscapes INT          Number of independent landscapes to evaluate on. Default 5.
    --min-burned INT          Minimum burned nodes in a no-treatment probe for an ignition to be
                              considered valid. Ignitions below this threshold are resampled.
                              Default 50 (2% of a 50x50 grid). Prevents natural fire extinction
                              from being mistaken for strategy effectiveness.
    --max-ignition-tries INT  Maximum attempts to find a valid ignition per landscape before
                              falling back to the best seen. Default 20.
    """
    parser.add_argument("--landscapes", type=int, default=5, help="Number of independent landscapes (default 5).")
    parser.add_argument(
        "--min-burned",
        type=int,
        default=50,
        help="Minimum burned nodes in no-treatment probe for ignition to be valid (default 50).",
    )
    parser.add_argument(
        "--max-ignition-tries",
        type=int,
        default=20,
        help="Max attempts to find a valid ignition per landscape (default 20).",
    )


def find_valid_ignition(
    graph,
    rng: np.random.Generator,
    max_steps: int,
    intervention_delay: int,
    min_burned: int,
    max_tries: int,
) -> tuple:
    """
    Sample ignition points until one produces meaningful fire spread without treatment.

    Runs a no-treatment probe at each candidate ignition. If the fire burns fewer than min_burned
    nodes it is discarded and a fresh ignition is sampled. If no valid ignition is found within
    max_tries attempts, the best seen so far is returned with a warning.
    """
    best_ignition = None
    best_burned = -1
    for attempt in range(max_tries):
        ignition = select_ignition_node(graph, rng)
        burned, _ = evaluate(
            lambda g, n: 0.0,
            graph,
            [ignition],
            0,
            max_steps,
            np.random.default_rng(rng.integers(2**31)),
            intervention_delay,
        )
        if burned >= min_burned:
            return ignition
        if burned > best_burned:
            best_burned = burned
            best_ignition = ignition
        log.debug("ignition %s burned %d < %d (attempt %d), resampling", ignition, burned, min_burned, attempt + 1)
    log.warning("No valid ignition found after %d tries (best burned=%d); using best available", max_tries, best_burned)
    return best_ignition
