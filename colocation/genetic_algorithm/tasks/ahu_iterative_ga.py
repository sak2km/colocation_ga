import logging
import numba as nb
import numpy as np
from ...utils import io
from ..optimizers import strict_genetic_algorithm as ga

Array = np.ndarray

LOG = logging.getLogger("M.GA.AHU_ITER")


def iterative_refine(
    ts_vav: Array,
    ts_ahu: Array,
    v_counts: Array,
    max_iteration: int,
    population_continuous: bool,
    ga_config: dict,
    mask_config: dict,
    cache_dir: str = None,
    **kwargs
):
    n_ahu = ts_ahu.shape[0]
    n_vav = ts_vav.shape[0]
    mask = np.ones((n_ahu, ts_ahu.shape[1]), dtype=bool)
    perfect_corr = perfect_corr_matrix(v_counts)
    accuracy_func = compile_vav_ahu_score(perfect_corr, v_counts)

    n_population = ga_config["n_population"]
    population = np.array(
        [np.random.permutation(n_vav) for _ in range(n_population)]
    ).reshape(n_population, -1, 1)

    for iteration in range(max_iteration):
        masked_ahu = ts_ahu * mask
        assert masked_ahu.shape == ts_ahu.shape

        corrcoef = pair_wise_correlation(masked_ahu, ts_vav)
        assert corrcoef.shape == (n_ahu, n_vav), str(corrcoef.shape)
        score_func = compile_vav_ahu_score(corrcoef, v_counts)

        v_sequence = run_ga(
            initial_population=population, score_func=score_func, **ga_config
        )
        # v_sequence = perfect_solution(v_counts)
        assert v_sequence.shape == (n_vav, 1)

        score = score_func(v_sequence)
        accuracy = accuracy_func(v_sequence)

        LOG.debug("Iteration %d: score=%f, accuracy=%f", iteration, score, accuracy)

        if cache_dir:
            io.save_variables(
                cache_dir, iteration, mask=mask, score=score, accuracy=accuracy, corr_matrix=corrcoef, solution=v_sequence
            )

        v_partitioned = list(partition_item(v_counts, v_sequence.reshape(-1)))
        assert len(v_partitioned) == n_ahu
        mask = make_mask(corrcoef, v_partitioned, ts_vav, **mask_config)
        if not population_continuous:
            population = np.array(
                [np.random.permutation(n_vav) for _ in range(n_population)]
            ).reshape(n_population, -1, 1)


def make_mask(
    corrcoef: Array,
    v_partitioned: list,
    v_ts: Array,
    k: int,
    maxpool_width: int,
    vote_threshold: int,
):
    n_ahu = corrcoef.shape[0]
    masks = np.empty((n_ahu, v_ts.shape[1]), dtype=bool)
    for a_id, vavs in enumerate(v_partitioned):
        top_k = vavs[np.argsort(corrcoef[a_id, vavs])[-k:]]
        assert top_k.shape == (k,)

        top_k_ts = v_ts[top_k]
        if maxpool_width != 0:
            top_k_ts = maxpool_1d(top_k_ts, maxpool_width)

        vote = np.sum(top_k_ts, axis=0)
        assert vote.shape == (v_ts.shape[1],)

        masks[a_id] = vote > vote_threshold
    return masks


def run_ga(
    initial_population: Array,
    score_func,
    n_population: int,
    n_survival: int,
    n_replaced: int,
    n_generation: int,
    mutation_rate: float,
):
    population = initial_population
    fitnesses = np.zeros(n_population)

    for generation in range(n_generation):
        fitnesses[:] = ga.fitness(population, score_func)

        _, survived, died = ga.survival_of_fittest(fitnesses, n_survival, n_replaced)

        if generation % 100 == 0:
            LOG.debug("GA Generation %d: %f", generation, np.max(fitnesses))

        population[:] = ga.next_gen(population, survived, died, 0, mutation_rate, None)

    fitnesses[:] = ga.fitness(population, score_func)

    best = np.argmax(fitnesses)

    return population[best]


# @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]))
def pair_wise_correlation(a: Array, b: Array):
    matrix = np.abs(np.corrcoef(a, b)[: a.shape[0], a.shape[0] :])

    if not np.isnan(matrix).any:
        return matrix

    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            matrix[i, j] = np.corrcoef(a[i], b[j])[0, 1]
    nans = np.isnan(matrix)
    matrix[np.where(nans)] = 0
    return matrix


def partition_item(counts, item):
    for i, j in partition(counts):
        yield item[i:j]


def partition(counts):
    start = 0

    for c in counts:
        yield start, start + c
        start += c


@nb.jit(nb.float64[:](nb.float64[:], nb.int32))
def maxpool_1d(data, width):
    pooled = np.zeros_like(data)
    col_ids = np.arange(data.shape[1])
    left_ids = np.clip(col_ids - width, a_min=0, a_max=None)
    right_ids = np.clip(col_ids + width + 1, a_max=col_ids.shape[0], a_min=None)

    for i in range(col_ids.shape[0]):
        pooled[:, col_ids[i]] = np.max(data[:, left_ids[i] : right_ids[i]], axis=1)

    return pooled


def compile_vav_ahu_score(corrcoef, v_counts):
    n_vav = corrcoef.shape[1]
    partitioned_ids = [(a_id, i, j) for a_id, (i, j) in enumerate(partition(v_counts))]

    # @nb.jit(nb.float64(nb.int32[:]))
    def wrapper(v_sequence: Array):
        v_sequence = v_sequence.reshape(-1)
        total_score = 0
        for a_id, i, j in partitioned_ids:
            total_score += np.sum(corrcoef[a_id, v_sequence[i:j]])
        return total_score / n_vav

    return wrapper


def perfect_solution(v_counts):
    return np.arange(np.sum(v_counts), dtype=int).reshape(-1, 1)


def perfect_corr_matrix(v_counts):
    matrix = np.zeros((len(v_counts), np.sum(v_counts)), dtype=float)
    for a_id, (i, j) in enumerate(partition(v_counts)):
        matrix[a_id, i:j] = 1
    return matrix
