"""Training scenario panels for the reinforcement learning environment.

Two ways of supplying training scenarios are supported:

**Fixed panel.** Reuse the exact landscapes a wildfireGP run trained on --- same grid seeds, same wind, same ignition
clusters --- so "identical training conditions" with a GP run is verifiable rather than asserted. :func:`load_gp_panel`
reconstructs the panel from a run's recorded ``config.json``.

**Per-episode sampling.** Draw a fresh landscape every episode from the same distribution (configured on
:class:`wildfireGP.rl.env.EnvConfig`). This gives a learner far more landscape variety than a fixed panel.

Why reconstruct rather than replay
----------------------------------
``scripts/run_gp.py`` derives ``land_seed = args.seed + i`` and then consumes a shared RNG for wind direction and for
the ignition search. Replaying that stream forward does not reproduce: the ignition search's RNG consumption depends
on how many candidate clusters it rejected, which is not recorded. The config *does* record the resulting winds and
ignition clusters verbatim, so reading them back is both exact and simpler.

Seed convention: training seeds 0-3, validation seed 42, held-out test seed 99. Seeds for per-episode sampling should
avoid all of them (:data:`RESERVED_SEEDS`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RESERVED_SEEDS = frozenset({0, 1, 2, 3, 42, 99})


@dataclass(frozen=True)
class ScenarioSpec:
    """One fully specified training scenario: which landscape, what weather, where it ignites."""

    landscape_seed: int
    wind_speed: float
    wind_direction: float
    ignition_nodes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PanelSpec:
    """A fixed panel of scenarios plus the landscape parameters they share."""

    scenarios: tuple[ScenarioSpec, ...]
    rows: int
    cols: int
    terrain_smoothing: float
    fuel_smoothing: float
    water_fraction: float
    rock_fraction: float
    fuel_moisture: float
    treatments_per_step: int
    max_steps: int
    intervention_delay: int
    min_treatment_distance: int
    source: str = ""

    def __len__(self) -> int:
        return len(self.scenarios)


def load_gp_panel(config_path: str | Path) -> PanelSpec:
    """Rebuild a wildfireGP training panel from a run's ``config.json``.

    :param config_path: Path to ``results/<run>/config.json`` in the wildfireGP repo.
    :return: The panel, with one :class:`ScenarioSpec` per landscape the GP trained on.
    """
    path = Path(config_path).expanduser()
    scenario = json.loads(path.read_text())["scenario"]
    base_seed = scenario["seed"]
    if base_seed is None:
        raise ValueError(f"{path} records seed=None; the landscapes are not reproducible.")

    winds = scenario["wind_per_landscape"]
    ignitions = scenario["ignition_nodes_per_landscape"]
    if not (len(winds) == len(ignitions) == scenario["landscapes"]):
        raise ValueError(
            f"{path} is inconsistent: landscapes={scenario['landscapes']}, "
            f"winds={len(winds)}, ignitions={len(ignitions)}."
        )

    scenarios = tuple(
        ScenarioSpec(
            landscape_seed=base_seed + i,
            wind_speed=float(winds[i]["speed"]),
            wind_direction=float(winds[i]["direction"]),
            ignition_nodes=tuple((int(r), int(c)) for r, c in ignitions[i]),
        )
        for i in range(scenario["landscapes"])
    )
    return PanelSpec(
        scenarios=scenarios,
        rows=scenario["rows"],
        cols=scenario["cols"],
        terrain_smoothing=scenario["terrain_smoothing"],
        fuel_smoothing=scenario["fuel_smoothing"],
        water_fraction=scenario["water_fraction"],
        rock_fraction=scenario["rock_fraction"],
        fuel_moisture=scenario["fuel_moisture"],
        treatments_per_step=scenario["treatments_per_step"],
        max_steps=scenario["max_steps"],
        intervention_delay=scenario["intervention_delay"],
        min_treatment_distance=scenario["min_treatment_distance"],
        source=str(path),
    )
