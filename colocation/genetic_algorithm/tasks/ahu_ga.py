"""Strict GA Task
"""
import numpy as np
from ..optimizers import strict_genetic_algorithm as ga
from ..optimizers import variable_genetic_algorithm as vga
from ..core import calcs
from ..data_loader import matrix_loader
from ..data_loader import config_loader
from ..utils import cache_dict
from ..core import corr_score
from ...utils import io
import numba
import pdb


import logging

LOG = logging.getLogger("M.VarGA")


def run(config: config_loader.ColocationConfig, vav_counts):
    """run a strict Genetic optimizer
    """
    # Prepare cache
    cache = cache_dict.get_cache(config)
    np.random.seed(config.seed)

    # Load matrix
    corr_matrix = matrix_loader.load_matrix(config.corr_matrix_path)


    # corr_matrix.shape = (24, 120)
    # config.room_count = 113
    # len(vav_counts) = 8
    # 121 ?
    # vav_counts = array([17, 4, 12, 12, 13, 19, 10, 26])
    
    # pdb.set_trace()
    assert corr_matrix.shape[0] == config.room_count + len(vav_counts)

    # If necessary, choose rooms
    if config.selected_rooms:
        config.selected_rooms = config.selected_rooms[: config.room_count]
        corr_matrix = matrix_loader.select_rooms(
            corr_matrix, config.selected_rooms, config.type_count
        )
        suppose_size = config.type_count * config.room_count
        assert corr_matrix.shape == (
            suppose_size,
            suppose_size,
        ), "Shape Error: {}".format(corr_matrix.shape)

    # Compile functions
    corr_func = _compile_ahu_fitness(corr_matrix, vav_counts)
    accuracy_func = _compile_ahu_fitness(perfect_matrix(vav_counts), vav_counts)
    assert np.max(perfect_matrix(vav_counts)) == 1.0
    assert np.min(perfect_matrix(vav_counts)) == 0.0
    assert np.sum(perfect_matrix(vav_counts)) == np.sum(vav_counts ** 2) + 2 * np.sum(
        vav_counts
    )

    # For testing
    _perfect_solution = np.arange(np.sum(vav_counts)).reshape(-1, 1) + len(vav_counts)

    assert (
        accuracy_func(_perfect_solution) == 1.0
    ), f"{accuracy_func(_perfect_solution)}"

    # For testing purpose:
    # corr_func = accuracy_func

    weight_func = (
        corr_score.compile_room_func(corr_matrix, config.type_count)
        if config.mutation_weighted
        else None
    )

    population = _initialize_population(config.population_count, vav_counts)
    best_fitness = 0
    best_solution = None
    mutation_rate = config.mutation_rate

    assert not np.isnan(population).any(), "population has nan"
    for iteration in range(config.max_iteration):
        fitnesses: np.ndarray = ga.fitness(population, corr_func)

        best, winners, losers = ga.survival_of_fittest(
            fitnesses, config.survivor_count, config.replaced_count
        )

        best_fitness = fitnesses[best]
        best_solution = np.copy(population[best])

        if iteration % 100 == 0 and config.verbose:
            io.vput(
                "Iteration [{}]: {}", iteration, fitnesses[best], verbose=config.verbose
            )

        if config.plot_fitness_density:
            cache["fitness_per_room"].append(fitnesses)
        if config.plot_fitness_accuracy:
            cache["best_fitness"].append(fitnesses[best])
            cache["accuracies"].append(accuracy_func(population[best]))

        population = ga.next_gen(
            population,
            winners,
            losers,
            config.crossing_over_rate,
            mutation_rate,
            weight_func=weight_func,
        )

        mutation_rate *= config.mutation_rate_decay

    if config.print_final_solution:
        print("Final Solution:")
        print(best_solution)

    for key, val in cache.items():
        io.save_npz(val, config.join_name("history_" + key + ".npz"))

    io.save_npz(best_solution, config.join_name("solution.npz"))

    io.vput("Final accuracy: {}", accuracy_func(best_solution), verbose=config.verbose)

    return best_fitness, accuracy_func(best_solution), cache


def _initialize_population(population_size, vav_counts):
    population = np.empty((population_size, np.sum(vav_counts)), dtype=int)
    perfect = np.arange(np.sum(vav_counts)) + len(vav_counts)

    for i in range(population.shape[0]):
        population[i] = np.copy(perfect)

    for i in range(population.shape[0]):
        np.random.shuffle(population[i])

    return population.reshape(population_size, np.sum(vav_counts), 1)


def perfect_matrix(vav_counts):
    """Perfect matrix
    """
    ahu_count = len(vav_counts)
    total_count = ahu_count + np.sum(vav_counts)
    matrix = np.zeros((total_count, total_count), dtype=float)

    start = ahu_count
    for a_id, v_count in enumerate(vav_counts):
        matrix[a_id, start : start + v_count].fill(1.0)
        matrix[start : start + v_count, a_id].fill(1.0)
        matrix[start : start + v_count, start : start + v_count].fill(1.0)

        start += v_count

    return matrix


def _compile_ahu_fitness(corr_matrix, vav_counts):
    @numba.jit(numba.float64(numba.int32[:, :]))
    def wrapper(solution):
        i = 0
        score = 0
        for a_id, v_count in enumerate(vav_counts):
            s_ids = solution[i : i + v_count].reshape(-1)
            score += np.mean(corr_matrix[a_id][s_ids])
            i += v_count
        score /= len(vav_counts)
        return score

    return wrapper
