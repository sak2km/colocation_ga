"""Strict GA Task
"""
import numpy as np
from ..optimizers import strict_genetic_algorithm as ga
from ..core import calcs
from ..data_loader import matrix_loader
from ..data_loader import config_loader
from ..utils import cache_dict
from ..core import corr_score
from ...utils import io


def run(config: config_loader.ColocationConfig):
    """run a strict Genetic optimizer

    Args:
        config (dict): configurations

    Returns:
        best fitness (float), accuracy of best solution, cache
    """
    # Prepare cache
    cache = cache_dict.get_cache(config)

    # Load matrix
    corr_matrix = matrix_loader.load_matrix(config.corr_matrix_path)

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
    corr_func = corr_score.compile_solution_func(corr_matrix, config.type_count)

    weight_func = (
        corr_score.compile_room_func(corr_matrix, config.type_count)
        if config.mutation_weighted
        else None
    )

    population = ga.initialize_population(
        config.population_count, config.room_count, config.type_count
    )
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
            cache["accuracies"].append(calcs.calculate_accuracy(population[best]))

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

    io.vput(
        "Final accuracy: {}",
        calcs.calculate_accuracy(best_solution),
        verbose=config.verbose,
    )

    return best_fitness, calcs.calculate_accuracy(best_solution), cache
