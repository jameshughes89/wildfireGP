"""
Shared CLI utilities for wildfireGP scripts.

add_landscape_args() registers the five simulation-scenario arguments that are common to run_gp.py,
compare_strategies.py, and animate.py. Centralising them here means defaults and help strings are defined once and all
scripts stay in sync automatically.
"""

import argparse
import subprocess


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
    """
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None = non-reproducible).")
    parser.add_argument("--rows", type=int, default=50, help="Landscape grid rows.")
    parser.add_argument("--cols", type=int, default=50, help="Landscape grid columns.")
    parser.add_argument("--treatments", type=int, default=3, help="Aggregate treatment budget per timestep.")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum simulation timesteps.")
    parser.add_argument(
        "--intervention-delay",
        type=int,
        default=3,
        help="Steps before treatments begin (models response time).",
    )


def run_formatters():
    for tool in ["isort .", "black .", "mdformat ."]:
        print(f"running `{tool}`")
        subprocess.run(tool, shell=True)


def run_verification():
    for tool in ["flake8 wildfireGP/ tests/", "codespell wildfireGP/ tests/"]:
        print(f"running `{tool}`")
        subprocess.run(tool, shell=True)
