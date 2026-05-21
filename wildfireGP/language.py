"""
DEAP typed primitive set for the wildfire GP language.

The GP tree is a function ``(state, node) -> float``, compiled and evaluated per node to produce a priority score for
treatment allocation. Higher score means higher priority.

PrimitiveSetTyped is used with two input types --- :class:`wildfireGP.network.SimState` and ``tuple`` --- and float
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
    burning_two_hop_count,
    distance_to_fire,
    elevation,
    elevation_delta_to_fire,
    fuel_level,
    fuel_moisture,
    has_treated_neighbour,
    mean_neighbour_elevation,
    mean_neighbour_fuel,
    reachable_unburned_area,
    slope,
    total_burned,
    total_burning,
    total_treated,
    total_unburned,
    treated_neighbour_count,
    unburnable_neighbour_count,
    unburned_neighbour_count,
    wind_fire_alignment,
    wind_speed,
)
from wildfireGP.network import SimState

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
    elevation,
    slope,
    mean_neighbour_elevation,
    mean_neighbour_fuel,
    burning_neighbour_count,
    burning_two_hop_count,
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

_GRAPH_FEATURES = [
    wind_speed,
    fuel_moisture,
    total_burning,
    total_burned,
    total_unburned,
    total_treated,
]


def _build() -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("MAIN", [SimState, tuple], float)
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
        pset.addPrimitive(func, [SimState, tuple], float, name=func.__name__)

    for func in _GRAPH_FEATURES:
        pset.addPrimitive(func, [SimState], float, name=func.__name__)

    pset.addEphemeralConstant("const", partial(random.uniform, -20.0, 20.0), float)

    return pset


PRIMITIVE_SET = _build()
