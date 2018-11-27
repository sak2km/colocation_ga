"""Variable genetic algorithm
"""
import numpy as np


def initialize_population(population_size, sensor_count, ahu_count):
    """Init populatin
    """
    population = np.array(
        [
            np.concatenate(
                (
                    np.arange(ahu_count),
                    np.random.randint(0, high=ahu_count, size=sensor_count - ahu_count),
                )
            ).reshape(1, -1)
            for _ in range(population_size)
        ]
    )
    return population


def corr_func(corr_matrix):
    def wrapper_func(labels):
        in_room = labels.T ^ np.repeat(labels, labels.shape[0], axis=0) == 0
        return corr_matrix * in_room / np.sum(in_room)

    return wrapper_func


def mutate(labels, mutation_rate, ahu_count):
    """mutation
    """
    mutate_mask = np.random.random(size=labels.shape[0]) < mutation_rate
    mutate_mask &= np.zeros(ahu_count)
    labels[mutate_mask] = np.random.randint(
        low=0, high=ahu_count, size=sum(mutate_mask)
    )


def next_gen(population, survivor_ids, loser_ids, mutation_rate, ahu_count):
    """Next gen
    """
    loser_count = len(loser_ids)
    mothers = np.random.choice(survivor_ids, size=loser_count)

    for i, l_id in enumerate(loser_ids):
        m_id = mothers[i]
        population[l_id] = population[m_id]

        mutate(population[l_id, 0], mutation_rate, ahu_count)

    return population


def accuracy(labels, ground_truth):
    return np.mean(labels.reshape(-1) == ground_truth)
