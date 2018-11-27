"""Running metric_learning
"""

import click
import metric_learn
import numpy as np

from ..utils import io
from ..utils import cli


def gen_distance_matrix(vectors):
    """Generate distance matrix
    """
    assert vectors.ndim == 2, "vectors must be 2-dimensioned"
    matrix = np.zeros((vectors.shape[0], vectors.shape[0]), dtype=float)

    for i, vec1 in enumerate(vectors):
        for j, vec2 in enumerate(vectors):
            matrix[i, j] = np.correlate(vec1, vec2)

    return matrix


_ML_MODEL = {
    "itml": metric_learn.ITML,
    "lfda": lambda *_: metric_learn.LFDA(k=7, dim=17),
    "lmnn": metric_learn.LMNN,
    "lsml": metric_learn.LSML,
    "sdml": metric_learn.SDML,
    "nca": metric_learn.NCA,
    "rca": metric_learn.RCA,
}


@click.command()
@click.option(
    "-a",
    "--algorithm",
    type=click.Choice(_ML_MODEL),
    help="Algorithm to use",
    required=True,
)
@cli.option_input_file()
@cli.option_output_dir()
@cli.option_verbose()
def run_learn(algorithm="", input_file="", output_dir="", verbose=0):
    """Run metric-learning algorithm on an input file
    """
    data = io.read_file(input_file)

    # TODO: a more general way of making labels
    labels = np.repeat(np.arange(50), 4)

    # TODO: tunable hyper pararmeter
    # TODO: use verbose
    model = _ML_MODEL[algorithm]()

    model.fit(data, labels)

    # TODO: more generic distance interface
    x_transformed = model.transform(data)
    print(x_transformed.shape)

    test_matrix = gen_distance_matrix(x_transformed)

    output_dir = io.make_dir(output_dir)
    io.save_npz(test_matrix, output_dir.joinpath("{}.npz".format(algorithm)))


if __name__ == "__main__":
    run_learn()
