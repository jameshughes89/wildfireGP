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

# Project CLI Scripts and Other Stuff

| Command  | Description                                    |
| -------- | ---------------------------------------------- |
| `format` | Automatically format Python and Markdown files |
| `verify` | Check for code style and spelling errors       |
| `pytest` | Run project unit tests                         |
