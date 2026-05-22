"""
DEAP GP engine for wildfire suppression strategy evolution.

Fitness is total_burned only (weights=(-1,)). Each individual is a GP tree compiled from PRIMITIVE_SET to a callable
(graph, node) -> float that scores nodes for treatment allocation. peak_burning and tree height are recorded in the
logbook for diagnostics but do not influence selection.

Re-evaluation every generation
-------------------------------
All individuals are re-evaluated every generation. The spread simulation is stochastic, so the same tree can produce
different fitness values across calls. Re-evaluating every generation prevents an individual from coasting on a lucky
draw from an earlier generation.

A single NumPy RNG is shared across all evaluations in a generation, so each individual is evaluated on a different
stochastic fire realization as the generator advances through the population. This makes within-generation fitness
comparisons noisier than if all individuals saw the same realization, but avoids the alternative problem of every
individual in a generation overfitting to one specific fire trajectory.

Tree generation
---------------
_gen_grow() is a type-aware replacement for DEAP's genGrow/genFull. The typed pset uses GraphState and tuple as
pass-through input types that carry no primitives --- only terminals (ARG0, ARG1). DEAP's generate() decides
terminal-vs-primitive without knowing the current type, so it will occasionally try to pick a primitive of type
GraphState and fail. _gen_grow() adds a single guard: if the current type has no primitives, always use a terminal.

Bloat control
-------------
staticLimit decorates mate and mutate with hard caps on both tree height and node count. Height alone allows wide
shallow trees; node count alone allows deep narrow trees. Both caps are needed to bound tree growth. Initial population
generation uses the same height cap to ensure the invariant holds from generation zero.

Randomness
----------
rng controls the spread simulation (NumPy). DEAP variation operators (cxOnePoint, mutUniform, _gen_grow) use Python's
random module. Call random.seed() before run() for full reproducibility of tree generation and variation.
"""

import logging
import operator
import random
import sys
from dataclasses import dataclass

import numpy as np
from deap import algorithms, base, creator, gp, tools
from deap.gp import MetaEphemeral

from wildfireGP.evaluate import (
    DEFAULT_INTERVENTION_DELAY,
    annotate_required_precomputes,
)
from wildfireGP.evaluate import evaluate as _sim_evaluate
from wildfireGP.language import PRIMITIVE_SET
from wildfireGP.network import GraphState

log = logging.getLogger(__name__)


@dataclass
class GPConfig:
    """
    Hyperparameters for the GP evolutionary loop.
    """

    population_size: int = 500
    generations: int = 100
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    tournament_size: int = 2
    max_tree_height: int = 5
    max_tree_nodes: int = 32
    elitism: int = 1
    init_min_height: int = 2
    init_max_height: int = 4
    mutation_min_height: int = 0
    mutation_max_height: int = 3


def build_toolbox(
    scenarios: list[tuple[GraphState, list[tuple]]],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    config: GPConfig,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> base.Toolbox:
    """
    Build and return a DEAP Toolbox wired to the given simulation scenarios.

    :param scenarios: List of (graph, ignition_nodes) pairs. Each individual is evaluated on all scenarios
        and the mean total_burned is reported as its fitness. Graphs are deepcopied internally per evaluation.
    :param treatments_per_step: Treatment budget per timestep.
    :param max_steps: Maximum simulation timesteps.
    :param rng: NumPy random generator for spread stochasticity.
    :param config: GP hyperparameters.
    :param intervention_delay: Steps before treatments begin. See evaluate() for full documentation.
    :return: Configured DEAP Toolbox.
    """
    _register_types()

    init_max_height = min(config.init_max_height, config.max_tree_height)

    toolbox = base.Toolbox()
    toolbox.register("expr", _gen_grow, pset=PRIMITIVE_SET, min_=config.init_min_height, max_=init_max_height)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=PRIMITIVE_SET)
    toolbox.register(
        "evaluate",
        _eval_individual,
        scenarios=scenarios,
        treatments_per_step=treatments_per_step,
        max_steps=max_steps,
        rng=rng,
        intervention_delay=intervention_delay,
    )
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", _gen_grow, min_=config.mutation_min_height, max_=config.mutation_max_height)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=PRIMITIVE_SET)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_tree_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_tree_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=config.max_tree_nodes))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=config.max_tree_nodes))

    return toolbox


def run(
    config: GPConfig,
    scenarios: list[tuple[GraphState, list[tuple]]],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
    hof_size: int = 5,
) -> tuple[list, tools.Logbook, tools.HallOfFame]:
    """
    Run the GP evolutionary loop and return the final population, statistics logbook, and hall of fame.

    The HOF tracks the best hof_size individuals seen across all generations, not just the final population. This
    prevents losing a good individual that was selected out due to evaluation noise in later generations.

    Retrieve and compile the best individual with:
        func = gp.compile(hof[0], pset=PRIMITIVE_SET)

    :param config: GP hyperparameters.
    :param scenarios: List of (graph, ignition_nodes) pairs. Each individual's fitness is the mean total_burned
        across all scenarios. Each graph must have wind and moisture set before calling.
    :param treatments_per_step: Treatment budget per simulation timestep.
    :param max_steps: Maximum simulation timesteps before forced termination.
    :param rng: NumPy random generator.
    :param intervention_delay: Steps before treatments begin. See evaluate() for full documentation.
    :param hof_size: Number of best individuals to track across all generations.
    :return: (population, logbook, hof). Logbook records gen, fitness (mean total_burned), size, peak_burning, and
        tree height stats for every generation. hof contains the best hof_size individuals seen during evolution.
    """
    toolbox = build_toolbox(scenarios, treatments_per_step, max_steps, rng, config, intervention_delay)
    mstats = _build_stats()
    logbook = tools.Logbook()
    logbook.header = ["gen", "fitness", "size", "peak", "height"]
    hof = tools.HallOfFame(hof_size)

    pop = toolbox.population(n=config.population_size)
    _evaluate_all(toolbox, pop)
    hof.update(pop)
    logbook.record(gen=0, **mstats.compile(pop))
    _log_gen(0, config.generations, pop)

    for gen in range(1, config.generations + 1):
        elites = list(map(toolbox.clone, tools.selBest(pop, config.elitism)))
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop) - config.elitism)))
        offspring = algorithms.varAnd(offspring, toolbox, config.crossover_prob, config.mutation_prob)
        pop[:] = elites + offspring
        _evaluate_all(toolbox, pop)
        hof.update(pop)
        logbook.record(gen=gen, **mstats.compile(pop))
        _log_gen(gen, config.generations, pop)

    return pop, logbook, hof


def _log_gen(gen: int, total_gens: int, pop: list) -> None:
    fit_vals = [ind.fitness.values[0] for ind in pop]
    log.info(
        "gen %d/%d  burned_min=%d  burned_avg=%.1f  peak_avg=%.1f  size_avg=%.1f  height_avg=%.1f",
        gen,
        total_gens,
        int(min(fit_vals)),
        float(np.mean(fit_vals)),
        float(np.mean([ind.peak_burning for ind in pop])),
        float(np.mean([len(ind) for ind in pop])),
        float(np.mean([ind.height for ind in pop])),
    )


def _gen_grow(pset, min_: int, max_: int, type_=None):
    """
    Type-aware grow generator for typed primitive sets with input-threading types.

    Identical to DEAP's genGrow except the terminal condition also fires when the current type has no
    primitives, preventing IndexError on types like GraphState and tuple that are terminals-only.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while stack:
        depth, type_ = stack.pop()
        if depth == height or not pset.primitives[type_] or (depth >= min_ and random.random() < pset.terminalRatio):
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, tb = sys.exc_info()
                raise IndexError(
                    f"_gen_grow tried to add a terminal of type '{type_}', but none is available."
                ).with_traceback(tb)
            if type(term) is MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, tb = sys.exc_info()
                raise IndexError(
                    f"_gen_grow tried to add a primitive of type '{type_}', but none is available."
                ).with_traceback(tb)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def _register_types() -> None:
    if not hasattr(creator, "FitnessWildfire"):
        creator.create("FitnessWildfire", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessWildfire)


def _eval_individual(
    individual,
    scenarios: list[tuple[GraphState, list[tuple]]],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    intervention_delay: int = DEFAULT_INTERVENTION_DELAY,
) -> tuple[float]:
    func = gp.compile(individual, pset=PRIMITIVE_SET)
    annotate_required_precomputes(func, {node.name for node in individual if getattr(node, "name", None) is not None})
    burns = []
    peaks = []
    for graph, ignition_nodes in scenarios:
        total_burned, peak_burning = _sim_evaluate(
            func, graph, ignition_nodes, treatments_per_step, max_steps, rng, intervention_delay
        )
        burns.append(total_burned)
        peaks.append(peak_burning)
    individual.peak_burning = float(np.mean(peaks))
    return (float(np.mean(burns)),)


def _evaluate_all(toolbox: base.Toolbox, population: list) -> None:
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)


def _build_stats() -> tools.MultiStatistics:
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    stats_peak = tools.Statistics(lambda ind: ind.peak_burning)
    stats_height = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, peak=stats_peak, height=stats_height)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats
