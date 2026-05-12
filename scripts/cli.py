"""
Shared CLI utilities for wildfireGP scripts.

add_landscape_args() registers the simulation-scenario arguments common to run_gp.py, compare_strategies.py, and
animate_simulation.py. Centralising them here means defaults and help strings are defined once and all scripts stay in
sync automatically.

load_candidate_by_expr() loads a compiled GP callable from a results directory by its expression string. Use this to
load a specific candidate identified from batch_evaluate.py output (--output CSV) rather than loading HOF .dill files
by index.
"""

import argparse
import pathlib

import dill

from wildfireGP.evaluate import DEFAULT_INTERVENTION_DELAY


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
