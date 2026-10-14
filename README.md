# wildfireGP

GP-evolved programs for wildfire suppression resource allocation on spatially-embedded graphs.

# Setup

## Bash

```sh
python3 -m venv --clear --prompt wildfireGP venv
. venv/bin/activate
pip install -e .
```

# Typical Workflow

Train a run with `python -m scripts.run_gp`, which writes a timestamped results directory under `results/`. Rank saved GP candidates across many landscapes with `python -m scripts.batch_evaluate --results-dir ...`, compare the strongest candidates against builtin baselines with `python -m scripts.compare_strategies --results-dir ...` or `--expr`, then use `plot_*` scripts and `animate_simulation` to inspect distributions, time series, spatial patterns, and example fire runs.

# Learned Policies (Reinforcement Learning Hooks)

`wildfireGP.rl` lets a learned policy (e.g. one trained with reinforcement learning) run on the same simulator, see the same 22 features as the GP language, and be scored by the same harness as the GP-evolved policies and baselines. It provides the hooks only; it does not include a network or trained models.

A learned policy is any function that maps the per-cell feature matrix, shape `(rows*cols, 22)` in the column order of `wildfireGP.rl.features.FEATURE_ORDER`, to one score per cell. Each timestep, the `K` highest-scoring eligible cells are treated, exactly as for a GP tree.

## Evaluating a learned policy

Requires only the core install.

```python
import numpy as np

from wildfireGP.evaluate import evaluate
from wildfireGP.network import create_grid, set_fuel_moisture, set_wind
from wildfireGP.rl.evaluation import as_scoring_function
from wildfireGP.rl.features import N_FEATURES

weights = np.random.default_rng(0).standard_normal(
    N_FEATURES
)  # stand-in for a trained model
policy = as_scoring_function(lambda features: features @ weights)

landscape = create_grid(rows=50, cols=50, seed=0)
set_wind(landscape, speed=36.0, direction=90.0)
set_fuel_moisture(landscape, moisture=0.2)
burned, peak = evaluate(
    policy, landscape, [(25, 25)], 3, 100, np.random.default_rng(0), 3, 0
)
```

`as_scoring_function` returns an ordinary strategy, so a learned policy can be passed anywhere a GP expression or baseline is accepted. Always measure learned policies this way, not through the training environment: `evaluate` breaks score ties at random, as it does for every other strategy, whereas a caller stepping the environment breaks them itself.

## Training a learned policy

Install the optional dependency, which adds [Gymnasium](https://gymnasium.farama.org/):

```sh
pip install -e ".[rl]"
```

`wildfireGP.rl.env.WildfireEnv` is a standard Gymnasium environment, usable with any RL library. Each step places one treatment, and the fire spreads once `K` treatments have been placed. Observations are the `(rows*cols, 22)` feature matrix, `action_masks()` marks the eligible cells (for action-masking algorithms such as MaskablePPO), and the reward is the negative number of cells burned by each spread step.

```python
from wildfireGP.rl.env import EnvConfig, WildfireEnv

env = WildfireEnv(
    EnvConfig(wind_speed_choices=(12.0, 18.0, 36.0, 48.0), wind_direction_random=True)
)
observation, info = env.reset(seed=1000)
```

`EnvConfig` defaults match the wildfireGP training conditions. To train on exactly the landscapes, winds, and ignitions of a recorded GP run, use `EnvConfig.from_panel(load_gp_panel("results/<run>/config.json"))`, with `load_gp_panel` from `wildfireGP.rl.scenarios`; otherwise a fresh landscape is sampled each episode, as above. Seeds 0-3, 42, and 99 are reserved for GP training, validation, and testing.

# Project CLI Scripts and Other Stuff

| Command  | Description                                    |
| -------- | ---------------------------------------------- |
| `format` | Automatically format Python and Markdown files |
| `verify` | Check for code style and spelling errors       |
| `pytest` | Run project unit tests                         |
