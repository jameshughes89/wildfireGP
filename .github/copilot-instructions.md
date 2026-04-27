# Copilot Instructions

## Project

GP-evolved programs for wildfire suppression resource allocation on spatially-embedded graphs. A tree-based GP program (via DEAP) takes per-node graph and state features as input and outputs a priority score for resource allocation. The goal is to minimize total burned area under resource constraints (K nodes treated per timestep).

This work generalizes prior epidemic mitigation research (eCov) to wildfire. The eCov repo on the Desktop is the reference implementation.

## Architecture

- `wildfireGP/spread.py` — probabilistic fire spread model (cellular automaton on graph)
- `wildfireGP/network.py` — landscape graph construction and node feature extraction
- `wildfireGP/language.py` — GP primitive set (terminals and functions)
- `wildfireGP/evaluate.py` — fitness evaluation (run spread simulation, score strategy)
- `wildfireGP/gp.py` — GP engine (DEAP setup, operators, main loop)
- `wildfireGP/strategies.py` — baseline heuristics for comparison

## Style

- Python 3.11+, line length 120
- Type hints on all function signatures
- No comments unless the WHY is non-obvious
- Tests mirror the module structure under `tests/`
- Property-based tests with Hypothesis where appropriate
- Formatting: black, isort, flake8
