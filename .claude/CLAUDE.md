# wildfireGP

## Project

GP-evolved programs for wildfire suppression resource allocation on spatially-embedded graphs. The core abstraction is a tree-based GP program that takes per-node graph/state features as input and outputs a priority score for resource allocation (firebreak placement, suppression deployment), with the goal of minimizing total burned area under resource constraints.

This work generalizes prior research (eCov, CIBCB 2021) from epidemic mitigation to wildfire. The eCov repo on the Desktop is the reference implementation — consult it for GP setup patterns, DEAP usage, and evaluation structure.

## Architecture (planned)

- `wildfireGP/spread.py` — probabilistic fire spread model (cellular automaton on graph)
- `wildfireGP/network.py` — landscape graph construction and node feature extraction
- `wildfireGP/language.py` — GP primitive set (terminals and functions)
- `wildfireGP/evaluate.py` — fitness evaluation (run spread simulation, score strategy)
- `wildfireGP/gp.py` — GP engine (DEAP setup, operators, main loop)
- `wildfireGP/strategies.py` — baseline heuristics for comparison

## Key design decisions

- Fire spread is a probabilistic cellular automaton: each burning node ignites neighbors with probability determined by fuel load, wind, and slope
- GP programs are tree-based, evaluated per-node to produce an allocation priority score
- Resource constraint: K nodes may be treated per timestep (analogous to vaccine budget in eCov)
- Fitness is multi-objective: minimize peak burning nodes AND total burned area

## Style

- Line length: 120
- Python 3.11+
- Type hints on all function signatures
- No comments unless the WHY is non-obvious
- Tests go in `tests/`, mirror the module structure
- Property-based tests with Hypothesis where appropriate
