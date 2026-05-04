"""
DEAP GP engine for wildfire suppression strategy evolution.

Fitness is (total_burned, peak_burning) — both minimised (weights=(-1, -1)). Each individual is a GP tree compiled from
PRIMITIVE_SET to a callable (graph, node) -> float that scores nodes for treatment allocation.

Re-evaluation every generation
-------------------------------
All individuals are re-evaluated every generation. The spread simulation is stochastic, so the same tree can produce
different fitness values across calls. Re-evaluating every generation prevents coasting on a lucky draw and ensures
fitness values within a generation are comparable.

Tree generation
---------------
_gen_grow() is a type-aware replacement for DEAP's genGrow/genFull. The typed pset uses nx.Graph and tuple as
pass-through input types that carry no primitives — only terminals (ARG0, ARG1). DEAP's generate() decides
terminal-vs-primitive without knowing the current type, so it will occasionally try to pick a primitive of type
nx.Graph and fail. _gen_grow() adds a single guard: if the current type has no primitives, always use a terminal.
This makes tree generation correct for input-threading types without changing any other behaviour.

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

import operator
import random
import sys
from dataclasses import dataclass

import networkx as nx
import numpy as np
from deap import algorithms, base, creator, gp, tools
from deap.gp import MetaEphemeral

from wildfireGP.evaluate import evaluate as _sim_evaluate
from wildfireGP.language import PRIMITIVE_SET


@dataclass
class GPConfig:
    """Hyperparameters for the GP evolutionary loop."""

    population_size: int = 100
    generations: int = 50
    crossover_prob: float = 0.8
    mutation_prob: float = 0.15
    tournament_size: int = 3
    max_tree_height: int = 5
    max_tree_nodes: int = 50
    elitism: int = 1


def build_toolbox(
    graph: nx.Graph,
    ignition_nodes: list[tuple],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
    config: GPConfig,
) -> base.Toolbox:
    """
    Build and return a DEAP Toolbox wired to the given simulation scenario.

    :param graph: Landscape template. Deepcopied internally per evaluation; the original is never modified.
    :param ignition_nodes: Nodes to ignite at t=0.
    :param treatments_per_step: Treatment budget per timestep.
    :param max_steps: Maximum simulation timesteps.
    :param rng: NumPy random generator for spread stochasticity.
    :param config: GP hyperparameters.
    :return: Configured DEAP Toolbox.
    """
    _register_types()

    init_max_height = min(4, config.max_tree_height)

    toolbox = base.Toolbox()
    toolbox.register("expr", _gen_grow, pset=PRIMITIVE_SET, min_=1, max_=init_max_height)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=PRIMITIVE_SET)
    toolbox.register(
        "evaluate",
        _eval_individual,
        graph=graph,
        ignition_nodes=ignition_nodes,
        treatments_per_step=treatments_per_step,
        max_steps=max_steps,
        rng=rng,
    )
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", _gen_grow, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=PRIMITIVE_SET)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_tree_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_tree_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=config.max_tree_nodes))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=config.max_tree_nodes))

    return toolbox


def run(
    config: GPConfig,
    graph: nx.Graph,
    ignition_nodes: list[tuple],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[list, tools.Logbook]:
    """
    Run the GP evolutionary loop and return the final population and statistics logbook.

    Retrieve and compile the best individual with:
        best = tools.selBest(population, 1)[0]
        func = gp.compile(best, pset=PRIMITIVE_SET)

    :param config: GP hyperparameters.
    :param graph: Landscape template. Wind and moisture must be set before calling.
    :param ignition_nodes: Nodes to ignite at t=0 for each evaluation.
    :param treatments_per_step: Treatment budget per simulation timestep.
    :param max_steps: Maximum simulation timesteps before forced termination.
    :param rng: NumPy random generator.
    :return: (population, logbook). Logbook records gen, fitness, and size stats for every generation.
    """
    toolbox = build_toolbox(graph, ignition_nodes, treatments_per_step, max_steps, rng, config)
    mstats = _build_stats()
    logbook = tools.Logbook()
    logbook.header = ["gen", "fitness", "size"]

    pop = toolbox.population(n=config.population_size)
    _evaluate_all(toolbox, pop)
    logbook.record(gen=0, **mstats.compile(pop))

    for gen in range(1, config.generations + 1):
        elites = list(map(toolbox.clone, tools.selBest(pop, config.elitism)))
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop) - config.elitism)))
        offspring = algorithms.varAnd(offspring, toolbox, config.crossover_prob, config.mutation_prob)
        pop[:] = elites + offspring
        _evaluate_all(toolbox, pop)
        logbook.record(gen=gen, **mstats.compile(pop))

    return pop, logbook


def _gen_grow(pset, min_: int, max_: int, type_=None):
    """
    Type-aware grow generator for typed primitive sets with input-threading types.

    Identical to DEAP's genGrow except the terminal condition also fires when the current type has no
    primitives, preventing IndexError on types like nx.Graph and tuple that are terminals-only.
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
        creator.create("FitnessWildfire", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessWildfire)


def _eval_individual(
    individual,
    graph: nx.Graph,
    ignition_nodes: list[tuple],
    treatments_per_step: int,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    func = gp.compile(individual, pset=PRIMITIVE_SET)
    return _sim_evaluate(func, graph, ignition_nodes, treatments_per_step, max_steps, rng)


def _evaluate_all(toolbox: base.Toolbox, population: list) -> None:
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)


def _build_stats() -> tools.MultiStatistics:
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean, axis=0)
    mstats.register("std", np.std, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)
    return mstats
