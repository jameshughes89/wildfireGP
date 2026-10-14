"""Gymnasium wrapper around the wildfireGP cellular-automaton simulator.

The step loop mirrors ``wildfireGP.evaluate.simulate`` exactly, so a policy trained here is measured on the same
mechanics the GP-evolved policies were evolved against. The simulator is read but not modified by this module.

Action decomposition
--------------------
The naive action space is 2500-way per timestep, choosing which cells to treat. Instead each ``step`` places a single
treatment, and the fire advances once ``treatments_per_step`` have accumulated. This decomposes the combinatorial
choice into K independent masked argmax picks over per-cell scores --- the same shape as the GP's
``f(state, cell) -> float`` interface, and the form action-masking RL algorithms (e.g. MaskablePPO) expect.

It also reproduces the GP's within-timestep mechanic. ``_apply_treatments`` rescores a cell's Moore neighbours after
each placement, because those are the only cells whose features can change; refreshing the whole observation between
placements yields identical values for every other cell, so the agent sees what a GP tree would.

One deliberate divergence: the four ``total_*`` graph features shift globally after each placement, and the GP's
incremental rescoring leaves them stale for non-neighbour cells. This env recomputes them. Recomputing is the correct
reading of the feature; the staleness is an artefact of the GP's neighbour-only rescore optimisation.

Reward
------
``-delta_burned`` for the spread step just completed, and 0.0 while treatments accumulate within a timestep. Dense,
so credit assignment does not have to bridge a whole episode. Episode return equals ``-(final_burned - burned_at_
first_decision_point)``; it differs from ``-total_burned`` by the pre-intervention burn, which no policy can affect.

Tie-breaking and evaluation
---------------------------
This env applies whatever cell the agent names; it does not select cells itself. A caller driving it with a scoring
function must therefore break ties itself, and ``argmax`` breaks them by lowest flat index --- which lays treatments
down in contiguous row-major runs. ``wildfireGP.evaluate._apply_treatments`` instead shuffles candidates before
sorting, so its ties break randomly.

For a continuous-valued scorer (an MLP) ties are measure-zero and the two agree. For the integer-scored doctrine
baselines they do not, and the gap is large: driving ``score_by_fire_proximity`` through this env pooled 852 burned
cells against the canonical harness's 1139, a 3x difference on one landscape. **Measure baselines with
``wildfireGP.evaluate.evaluate``, never through this env**, or the comparison hands an unearned advantage to
whichever side was measured here.

Termination
-----------
``terminated`` when no cell is burning (the fire is out --- a genuine MDP terminal). ``truncated`` when ``max_steps``
is reached with the fire still going. This matches ``simulate``'s loop, which breaks on the former and exhausts its
range on the latter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wildfireGP.evaluate import annotate_required_precomputes, evaluate, init_ignition
from wildfireGP.features import update_neighbourhood_maps_after_treatment
from wildfireGP.network import (
    GraphState,
    NodeState,
    create_grid,
    select_ignition_cluster,
    set_fuel_moisture,
    set_wind,
)
from wildfireGP.rl.features import (
    N_FEATURES,
    compute_action_mask,
    extract_features,
    run_precomputes,
)
from wildfireGP.rl.scenarios import PanelSpec, ScenarioSpec
from wildfireGP.spread import spread_step

MAX_SCENARIO_TRIES = 10


@dataclass(frozen=True)
class EnvConfig:
    """Simulation parameters. Defaults match the wildfireGP training conditions.

    Departures from these defaults invalidate the head-to-head comparison against the GP policies.
    """

    rows: int = 50
    cols: int = 50
    terrain_smoothing: float = 3.0
    fuel_smoothing: float = 3.0
    water_fraction: float = 0.0
    rock_fraction: float = 0.0
    cell_size_m: float = 100.0
    wind_speed: float = 20.0
    wind_direction: float = 0.0
    fuel_moisture: float = 0.2
    treatments_per_step: int = 3
    max_steps: int = 100
    intervention_delay: int = 3
    min_treatment_distance: int = 0
    ignition_cluster_size: int = 3
    min_burned: int = 50
    max_ignition_tries: int = 20

    # -- scenario sampling --------------------------------------------------
    # Fixed panel: typically wildfireGP.rl.scenarios.load_gp_panel of a wildfireGP run. One scenario is drawn
    # uniformly per episode, so coverage of the panel is balanced in expectation, matching the GP's evaluation of
    # every candidate on every landscape. Ignition clusters come from the panel and skip the fizzle filter, having
    # already been validated by the original run.
    panel: PanelSpec | None = None
    # Per-episode sampling: draw weather per episode. wind_speed_choices draws uniformly from the given set (the GP's
    # control cell used {12, 18, 36, 48} km/h); wind_direction_random draws uniform [0, 360). Both fall back to the
    # fixed wind_speed / wind_direction above when unset. Direction is held constant for the whole episode either way,
    # matching the simulator, where only set_wind ever writes it.
    wind_speed_choices: tuple[float, ...] | None = None
    wind_direction_random: bool = False

    @classmethod
    def from_panel(cls, panel: PanelSpec, **overrides) -> EnvConfig:
        """Build a config whose simulation parameters are taken from a recorded wildfireGP run."""
        return cls(
            rows=panel.rows,
            cols=panel.cols,
            terrain_smoothing=panel.terrain_smoothing,
            fuel_smoothing=panel.fuel_smoothing,
            water_fraction=panel.water_fraction,
            rock_fraction=panel.rock_fraction,
            fuel_moisture=panel.fuel_moisture,
            treatments_per_step=panel.treatments_per_step,
            max_steps=panel.max_steps,
            intervention_delay=panel.intervention_delay,
            min_treatment_distance=panel.min_treatment_distance,
            panel=panel,
            **overrides,
        )


# Zero-budget probe for the fizzle filter: no treatments are placed, so no feature is ever read and every precompute
# is wasted work. Annotating an empty set skips them without altering RNG consumption or fire behaviour.
_NO_TREATMENT_PROBE = annotate_required_precomputes(lambda state, node: 0.0, [])


class WildfireEnv(gym.Env):
    """Single-treatment-per-step wildfire suppression environment.

    Observation: ``(rows*cols, 22)`` float32, per-cell features in :data:`wildfireGP.rl.features.FEATURE_ORDER`, for
    every cell in row-major order. Guaranteed finite.

    Action: flat cell index in ``[0, rows*cols)``. Only cells where :meth:`action_masks` is True may be passed;
    anything else raises. MaskablePPO reads that mask via ``sb3_contrib``'s ``get_action_masks``.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config if config is not None else EnvConfig()
        n_cells = self.cfg.rows * self.cfg.cols
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_cells, N_FEATURES), dtype=np.float32)
        self.action_space = spaces.Discrete(n_cells)

        self.state: GraphState | None = None
        self.scenario: ScenarioSpec | None = None
        self.ignition_nodes: list[tuple] = []
        self._sim_rng = np.random.default_rng()
        self._sim_step = 0
        self._treatments_left = 0
        self._episode_over = False
        self._truncated = False
        self._peak_burning = 0
        self._features = np.zeros((n_cells, N_FEATURES), dtype=np.float32)
        self._mask = np.zeros(n_cells, dtype=bool)

    # -- gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Start an episode.

        ``options`` may inject a fixed scenario instead of sampling one, which is how an evaluation harness
        can run a test panel on the same landscapes the GP policies were scored on:

        - ``state``: a :class:`GraphState` template. Copied internally; the caller's object is not mutated.
        - ``ignition_nodes``: cells to ignite at t=0. Bypasses the fizzle filter.
        - ``sim_seed``: seed for the spread RNG. Defaults to a draw from this env's seeded generator.
        """
        super().reset(seed=seed)
        if options is not None and "state" in options:
            self._set_scenario(
                options["state"].copy(),
                list(options["ignition_nodes"]),
                options.get("sim_seed"),
            )
            self._advance_to_decision_point()
            if self._episode_over:
                raise RuntimeError(
                    "Injected scenario ended during the intervention delay; it leaves no action to take."
                )
            return self._features, self._info()
        for _ in range(MAX_SCENARIO_TRIES):
            self._new_scenario()
            self._advance_to_decision_point()
            if not self._episode_over:
                return self._features, self._info()
        raise RuntimeError(
            f"No scenario survived the {self.cfg.intervention_delay}-step intervention delay in "
            f"{MAX_SCENARIO_TRIES} attempts; check EnvConfig.min_burned and the landscape parameters."
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._episode_over:
            raise RuntimeError("step() called on a finished episode; call reset() first.")
        action = int(action)
        if not self._mask[action]:
            raise ValueError(f"Action {action} is masked out; sample only from action_masks().")

        self._place_treatment(divmod(action, self.cfg.cols))
        self._treatments_left -= 1
        self._mask = compute_action_mask(self.state, self.cfg.min_treatment_distance)

        if self._treatments_left > 0 and self._mask.any():
            self._features = extract_features(self.state)
            return self._features, 0.0, False, False, self._info()

        burned_before = self._burned_count()
        self._spread()
        reward = -float(self._burned_count() - burned_before)

        self._advance_to_decision_point()
        terminated = self._episode_over and not self._truncated
        truncated = self._episode_over and self._truncated
        return self._features, reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """Return the ``(rows*cols,)`` bool mask of legal actions. Read by MaskablePPO each step."""
        return self._mask.copy()

    # -- internals ----------------------------------------------------------

    def _new_scenario(self) -> None:
        cfg = self.cfg
        spec = self._draw_scenario()
        state = create_grid(
            rows=cfg.rows,
            cols=cfg.cols,
            terrain_smoothing=cfg.terrain_smoothing,
            fuel_smoothing=cfg.fuel_smoothing,
            water_fraction=cfg.water_fraction,
            rock_fraction=cfg.rock_fraction,
            cell_size_m=cfg.cell_size_m,
            seed=spec.landscape_seed,
        )
        set_wind(state, speed=spec.wind_speed, direction=spec.wind_direction)
        set_fuel_moisture(state, moisture=cfg.fuel_moisture)

        ignition = list(spec.ignition_nodes) if spec.ignition_nodes else self._find_valid_ignition(state)
        self.scenario = spec
        self._set_scenario(state, ignition, None)

    def _draw_scenario(self) -> ScenarioSpec:
        """Pick this episode's landscape and weather, from the fixed panel or by sampling."""
        cfg = self.cfg
        if cfg.panel is not None:
            return cfg.panel.scenarios[int(self.np_random.integers(len(cfg.panel)))]
        speed = (
            float(cfg.wind_speed_choices[int(self.np_random.integers(len(cfg.wind_speed_choices)))])
            if cfg.wind_speed_choices
            else cfg.wind_speed
        )
        direction = float(self.np_random.uniform(0.0, 360.0)) if cfg.wind_direction_random else cfg.wind_direction
        return ScenarioSpec(
            landscape_seed=int(self.np_random.integers(2**31)),
            wind_speed=speed,
            wind_direction=direction,
            ignition_nodes=(),
        )

    def _set_scenario(self, state: GraphState, ignition_nodes: list[tuple], sim_seed: int | None) -> None:
        init_ignition(state, ignition_nodes)
        self.state = state
        self.ignition_nodes = ignition_nodes
        self._sim_rng = np.random.default_rng(
            int(self.np_random.integers(2**31)) if sim_seed is None else int(sim_seed)
        )
        self._sim_step = 0
        self._treatments_left = 0
        self._episode_over = False
        self._truncated = False
        self._peak_burning = int((state.state == NodeState.BURNING).sum())

    def _find_valid_ignition(self, state: GraphState) -> list[tuple]:
        """Sample ignition clusters until one burns >= min_burned cells untreated.

        Mirrors ``scripts.cli.find_valid_ignition_cluster`` so training ignitions match the fizzle-filter discipline
        the GP training and evaluation scripts already apply. Without it a share of episodes would be fires that die
        on their own, which teaches the policy nothing and dilutes the reward signal.
        """
        cfg = self.cfg
        probe_rng = np.random.default_rng(int(self.np_random.integers(2**31)))
        best_cluster: list[tuple] | None = None
        best_burned = -1
        for _ in range(cfg.max_ignition_tries):
            cluster = select_ignition_cluster(state, probe_rng, size=cfg.ignition_cluster_size)
            burned, _ = evaluate(
                _NO_TREATMENT_PROBE,
                state,
                cluster,
                0,
                cfg.max_steps,
                np.random.default_rng(int(probe_rng.integers(2**31))),
                cfg.intervention_delay,
            )
            if burned >= cfg.min_burned:
                return cluster
            if burned > best_burned:
                best_burned = burned
                best_cluster = cluster
        return best_cluster if best_cluster is not None else []

    def _advance_to_decision_point(self) -> None:
        """Run spread steps until a treatment can be placed, or the episode ends.

        Mirrors ``simulate``: no treatments before ``intervention_delay``, and a timestep with no eligible cells
        simply spreads (the GP places ``min(budget, len(candidates))`` treatments, which is zero in that case).
        """
        cfg = self.cfg
        while True:
            if self._sim_step >= cfg.max_steps:
                self._finish(truncated=True)
                return
            if not (self.state.state == NodeState.BURNING).any():
                self._finish(truncated=False)
                return
            if self._sim_step < cfg.intervention_delay:
                self._spread()
                continue
            run_precomputes(self.state)
            self._mask = compute_action_mask(self.state, cfg.min_treatment_distance)
            if not self._mask.any():
                self._spread()
                continue
            self._treatments_left = cfg.treatments_per_step
            self._features = extract_features(self.state)
            return

    def _finish(self, truncated: bool) -> None:
        self._episode_over = True
        self._truncated = truncated
        self._treatments_left = 0
        run_precomputes(self.state)
        self._mask = compute_action_mask(self.state, self.cfg.min_treatment_distance)
        self._features = extract_features(self.state)

    def _place_treatment(self, node: tuple) -> None:
        """Apply one treatment, mirroring the per-placement bookkeeping in ``evaluate._apply_treatments``."""
        state = self.state
        state.state[node] = NodeState.TREATED
        state.step_treated += 1
        state.step_unburned -= 1
        update_neighbourhood_maps_after_treatment(state, node)

    def _spread(self) -> None:
        spread_step(self.state, self._sim_rng)
        self._sim_step += 1
        self._peak_burning = max(self._peak_burning, int((self.state.state == NodeState.BURNING).sum()))

    def _burned_count(self) -> int:
        return int((self.state.state == NodeState.BURNED).sum())

    def _info(self) -> dict:
        return {
            "landscape_seed": self.scenario.landscape_seed if self.scenario is not None else -1,
            "wind_speed": self.scenario.wind_speed if self.scenario is not None else 0.0,
            "sim_step": self._sim_step,
            "total_burned": self._burned_count(),
            "peak_burning": self._peak_burning,
            "treatments_left": self._treatments_left,
            "n_eligible": int(self._mask.sum()),
        }
