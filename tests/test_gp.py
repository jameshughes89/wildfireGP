import random

import numpy as np
import pytest
from deap import creator, gp

from wildfireGP.gp import (
    GPConfig,
    _eval_individual,
    _gen_grow,
    _register_types,
    build_toolbox,
    run,
)
from wildfireGP.language import PRIMITIVE_SET
from wildfireGP.network import create_grid, set_fuel_moisture, set_wind


def _graph():
    g = create_grid(5, 5, seed=0)
    set_wind(g, speed=10.0, direction=0.0)
    set_fuel_moisture(g, moisture=0.2)
    return g


def _config():
    return GPConfig(
        population_size=6, generations=2, tournament_size=2, max_tree_height=4, max_tree_nodes=20, init_max_height=3
    )


def _rng():
    return np.random.default_rng(0)


@pytest.fixture(autouse=True)
def seed_python_random():
    random.seed(42)


def test_fitness_weights_are_minimising():
    _register_types()
    assert creator.FitnessWildfire.weights == (-1.0,)


def test_run_population_size_preserved():
    config = _config()
    pop, _, _ = run(config, [(_graph(), [(2, 2)])], 2, 10, _rng())
    assert len(pop) == config.population_size


def test_run_logbook_records_all_generations():
    config = _config()
    _, log, _ = run(config, [(_graph(), [(2, 2)])], 2, 10, _rng())
    assert len(log) == config.generations + 1


def test_run_all_individuals_have_valid_fitness():
    pop, _, _ = run(_config(), [(_graph(), [(2, 2)])], 2, 10, _rng())
    assert all(ind.fitness.valid for ind in pop)


def test_run_all_individuals_obey_height_limit():
    config = _config()
    pop, _, _ = run(config, [(_graph(), [(2, 2)])], 2, 10, _rng())
    assert all(ind.height <= config.max_tree_height for ind in pop)


def test_run_all_individuals_obey_size_limit():
    config = _config()
    pop, _, _ = run(config, [(_graph(), [(2, 2)])], 2, 10, _rng())
    assert all(len(ind) <= config.max_tree_nodes for ind in pop)


def test_run_fitness_values_are_non_negative():
    pop, _, _ = run(_config(), [(_graph(), [(2, 2)])], 2, 10, _rng())
    for ind in pop:
        (total_burned,) = ind.fitness.values
        assert total_burned >= 0
        assert ind.peak_burning >= 0


def test_run_hof_size_bounded():
    hof_size = 3
    _, _, hof = run(_config(), [(_graph(), [(2, 2)])], 2, 10, _rng(), hof_size=hof_size)
    assert len(hof) <= hof_size


def test_run_hof_individuals_have_valid_fitness():
    _, _, hof = run(_config(), [(_graph(), [(2, 2)])], 2, 10, _rng())
    assert all(ind.fitness.valid for ind in hof)


def test_build_toolbox_reuses_registered_types_on_second_call():
    build_toolbox([(_graph(), [(2, 2)])], 2, 10, _rng(), _config())
    fitness_cls = creator.FitnessWildfire
    build_toolbox([(_graph(), [(2, 2)])], 2, 10, _rng(), _config())
    assert creator.FitnessWildfire is fitness_cls


def test_gen_grow_terminal_only_types_use_argument_terminals():
    expr_graph = _gen_grow(PRIMITIVE_SET, 1, 1, type_=type(_graph()))
    assert len(expr_graph) == 1
    assert isinstance(expr_graph[0], gp.Terminal)
    assert expr_graph[0].name == "ARG0"
    assert expr_graph[0].ret is type(_graph())

    expr_node = _gen_grow(PRIMITIVE_SET, 1, 1, type_=tuple)
    assert len(expr_node) == 1
    assert isinstance(expr_node[0], gp.Terminal)
    assert expr_node[0].name == "ARG1"
    assert expr_node[0].ret is tuple


def test_eval_individual_annotates_compiled_precompute_requirements():
    expr = gp.PrimitiveTree.from_string("add(distance_to_fire(graph, node), fuel_level(graph, node))", PRIMITIVE_SET)
    individual = creator.Individual(expr)
    result = _eval_individual(individual, [(_graph(), [(2, 2)])], 2, 5, _rng())
    assert result[0] >= 0.0
