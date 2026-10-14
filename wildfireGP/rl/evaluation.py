"""Evaluate a learned per-cell scorer through the canonical wildfireGP harness.

A learned policy is any callable that maps the ``(rows*cols, 22)`` feature matrix from
:func:`wildfireGP.rl.features.extract_features` to one score per cell. :func:`as_scoring_function` wraps it as the
``f(state, node) -> float`` strategy that ``wildfireGP.evaluate`` expects, so it is measured exactly as the GP-evolved
policies and the hand-coded baselines are.

Why not evaluate in the env
---------------------------
``wildfireGP.rl.env.WildfireEnv`` applies whatever cell the caller names, so the caller breaks ties --- and ``argmax``
breaks them by lowest flat index, which lays treatments in contiguous row-major runs. ``wildfireGP.evaluate`` shuffles
candidates instead. For the integer-scored doctrine baselines that difference is worth ~25% of pooled burned area, so
measuring a learned policy one way and the baselines the other would corrupt the comparison. Everything is therefore
scored through ``wildfireGP.evaluate.evaluate``, exactly as the GP policies are.

Making that efficient
---------------------
``_apply_treatments`` calls ``func(state, node)`` once per candidate --- ~2000 calls per timestep, plus a rescore of
the treated cell's neighbours after each placement. Calling a learned scorer per cell would mean ~200,000 forward
passes per simulation. :func:`as_scoring_function` instead scores every cell in one batched call and serves the cached
grid, recomputing only when the state has actually changed. The cache key is the treated/burned/burning/unburned counts
together with the object identity of the state, which is exactly what ``_apply_treatments`` mutates as it places
treatments and what the per-step precomputes refresh.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wildfireGP.evaluate import annotate_required_precomputes
from wildfireGP.network import GraphState
from wildfireGP.rl.features import FEATURE_ORDER, extract_features

CellScorer = Callable[[np.ndarray], np.ndarray]
"""A learned policy: maps the ``(rows*cols, 22)`` feature matrix to a ``(rows*cols,)`` array of scores."""


def _state_fingerprint(state: GraphState) -> tuple:
    counts = np.bincount(state.state.ravel(), minlength=4)
    return (id(state), int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]))


def as_scoring_function(scorer: CellScorer) -> Callable[[GraphState, tuple], float]:
    """Wrap a learned per-cell scorer as the ``f(state, node) -> float`` callable the wildfireGP harness expects.

    The returned function is annotated with the precomputes the 22 features need, so ``simulate`` runs exactly the
    same precompute set it would for a GP tree consuming the full language.

    :param scorer: Callable mapping the feature matrix to one score per cell, in the row-major cell order of
        :func:`wildfireGP.rl.features.extract_features`.
    :return: Strategy function usable anywhere a GP tree or hand-coded baseline is.
    """
    cache: dict = {"key": None, "scores": None}

    def score(state: GraphState, node: tuple) -> float:
        key = _state_fingerprint(state)
        if key != cache["key"]:
            cache["key"] = key
            cache["scores"] = np.asarray(scorer(extract_features(state)), dtype=np.float64).ravel()
        return float(cache["scores"][node[0] * state.cols + node[1]])

    return annotate_required_precomputes(score, FEATURE_ORDER)
