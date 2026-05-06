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
    --seed INT        Random seed. Default None draws from OS entropy (non-reproducible).
                      Pass an integer for fully reproducible runs.
    --rows INT        Grid rows. Default 20.
    --cols INT        Grid columns. Default 20.
    --treatments INT  Treatment budget per simulation timestep. Default 3.
    --max-steps INT   Maximum simulation timesteps before forced termination. Default 50.
    """
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None = non-reproducible).")
    parser.add_argument("--rows", type=int, default=20, help="Landscape grid rows.")
    parser.add_argument("--cols", type=int, default=20, help="Landscape grid columns.")
    parser.add_argument("--treatments", type=int, default=3, help="Treatment budget per timestep.")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum simulation timesteps.")


def run_formatters():
    for tool in ["isort .", "black .", "mdformat ."]:
        print(f"running `{tool}`")
        subprocess.run(tool, shell=True)


def run_verification():
    for tool in ["flake8 wildfireGP/ tests/", "codespell wildfireGP/ tests/"]:
        print(f"running `{tool}`")
        subprocess.run(tool, shell=True)
