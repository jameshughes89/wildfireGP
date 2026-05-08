"""
Developer tooling entry points (format and verify).

These are registered as console scripts in pyproject.toml and are not part of the simulation pipeline.
"""

import subprocess


def run_formatters():
    for tool in ["isort .", "black .", "mdformat ."]:
        print(f"running `{tool}`")
        subprocess.run(tool, shell=True)


def run_verification():
    for tool in ["flake8 wildfireGP/ scripts/ tests/", "codespell wildfireGP/ scripts/ tests/"]:
        print(f"running `{tool}`")
        subprocess.run(tool, shell=True)
