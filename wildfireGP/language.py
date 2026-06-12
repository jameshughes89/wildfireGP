"""DEAP typed primitive set for the wildfire GP language.

The GP tree is a function ``(state, node) -> float``, compiled and evaluated per node to produce a priority score for
treatment allocation. Higher score means higher priority.

PrimitiveSetTyped is used with two input types --- :class:`wildfireGP.network.GraphState` and ``tuple`` --- and float
output. Feature functions appear as branch nodes (primitives) rather than zero-arity terminals: they receive state
and/or node directly, threaded down from the tree root. The compiled tree is callable as ``func(state, node)``.

For backwards compatibility with previously saved ``.expr`` strings, the argument names remain ``graph`` and ``node``.

Arithmetic primitives operate on float only. The type system prevents the tree generator from passing a state or node
where a float is expected.

Infinite-valued features (``distance_to_fire``, ``burnable_distance_to_fire`` when no fire is present) propagate
through arithmetic normally. Clamping of the final priority score is the responsibility of the evaluator.
"""

import operator
import random
from functools import partial

from deap import gp

from wildfireGP.features import (
    burnable_distance_to_fire,
    burning_neighbour_count,
    distance_to_fire,
    elevation_delta_to_fire,
    fuel_level,
    has_treated_neighbour,
    mean_neighbour_fuel,
    reachable_unburned_area,
    slope,
    treated_neighbour_count,
    unburnable_neighbour_count,
    unburned_neighbour_count,
    wind_fire_alignment,
)
from wildfireGP.network import GraphState

# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def _protected_div(a: float, b: float) -> float:
    return a / b if b != 0.0 else 1.0


def _if_positive(condition: float, a: float, b: float) -> float:
    return a if condition > 0.0 else b


def _max2(a: float, b: float) -> float:
    return max(a, b)


def _min2(a: float, b: float) -> float:
    return min(a, b)


# ---------------------------------------------------------------------------
# Primitive set
# ---------------------------------------------------------------------------

_NODE_FEATURES = [
    fuel_level,
    slope,
    mean_neighbour_fuel,
    burning_neighbour_count,
    unburned_neighbour_count,
    unburnable_neighbour_count,
    has_treated_neighbour,
    treated_neighbour_count,
    distance_to_fire,
    burnable_distance_to_fire,
    reachable_unburned_area,
    wind_fire_alignment,
    elevation_delta_to_fire,
]


def _build() -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("MAIN", [GraphState, tuple], float)
    pset.renameArguments(ARG0="graph", ARG1="node")

    pset.addPrimitive(operator.add, [float, float], float, name="add")
    pset.addPrimitive(operator.sub, [float, float], float, name="sub")
    pset.addPrimitive(operator.mul, [float, float], float, name="mul")
    pset.addPrimitive(_protected_div, [float, float], float, name="div")
    pset.addPrimitive(operator.neg, [float], float, name="neg")
    pset.addPrimitive(_max2, [float, float], float, name="max")
    pset.addPrimitive(_min2, [float, float], float, name="min")
    pset.addPrimitive(_if_positive, [float, float, float], float, name="if_pos")

    for func in _NODE_FEATURES:
        pset.addPrimitive(func, [GraphState, tuple], float, name=func.__name__)

    pset.addEphemeralConstant("const", partial(random.uniform, -20.0, 20.0), float)

    return pset


PRIMITIVE_SET = _build()
