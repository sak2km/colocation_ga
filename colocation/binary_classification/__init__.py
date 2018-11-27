"""Binary Classification Task: Same Room Or Not
"""

import click
import numpy as np
from sklearn import linear_model, model_selection, svm
from ..utils import io, cli


def prepare_pair(data: np.ndarray, type_count: int):
    """pair up data
    """
    sensor_count, feature_count = data.shape
    pair_count = sensor_count * (sensor_count - 1)
    pair_data = np.zeros((pair_count, feature_count ** 2))
    pair_label = np.zeros((pair_count), dtype=int)

    k = 0
    for i in range(sensor_count):
        for j in range(sensor_count):
            if i == j:
                continue
            pair_data[k] = np.outer(data[i].T, data[j]).reshape(-1)
            pair_label[k] = 1 if i // type_count == j // type_count else 0
            k += 1

    return pair_data, pair_label


def unbias_training_data(data, labels):
    """Even-ize data
    """
    sample_count = labels.shape[0]
    ones = np.where(labels == 1)
    zeros = np.where(labels == 0)

    weights = np.zeros(sample_count)
    weights[ones] = len(zeros[0])
    weights[zeros] = len(ones[0])
    probability = weights / np.sum(weights)

    even_ids = np.random.choice(
        np.arange(sample_count), size=2 * sample_count, p=probability
    )
    even_data = data[even_ids]
    even_label = labels[even_ids]

    return even_data, even_label


def build_coef_matrix(coefs, sensor_count):
    """Build coefficient matrix be unrolling the coeffients
    """
    matrix = np.zeros((sensor_count, sensor_count))

    coef_iter = iter(coefs)
    for i in range(sensor_count):
        for j in range(sensor_count):
            matrix[i, j] = next(coef_iter) if i != j else 1
    return matrix


class NoModel(object):
    """NOOP-ish class
    """

    def fit(self, *_):
        """Do nothing
        """
        return self

    def score(self, _, y):
        """Return randomly generated labels
        """
        random_labels = np.zeros_like(y)
        return np.sum(random_labels == y) / y.shape[0]

    def predict_proba(self, x):
        """Return randomly generated labels
        """
        random_labels = np.random.randint(0, 2, size=x.shape[0])
        return random_labels


def _select_model(model_name, verbose):
    if model_name == "log-regression":
        return linear_model.LogisticRegression(
            verbose=verbose, class_weight="balanced", max_iter=99999999, solver="saga"
        )
    elif model_name == "svc":
        return svm.SVC(class_weight="balanced", verbose=verbose, probability=True)
    elif model_name == "no-model":
        return NoModel()
    else:
        raise KeyError("Model name not found")


def _evaluate(predictions, labels):
    result = np.zeros((2, 2))
    for pred, label in zip(predictions, labels):
        result[label, pred] += 1

    result /= labels.size
    return result


@click.command("classify")
@click.argument("model", type=click.Choice(["log-regression", "svc", "no-model"]))
@cli.option_input_file()
@cli.option_output_file()
@click.option("--type-count", default=4)
@click.option("--train-fraction", default=0.75)
@click.option("--seed", default=0)
@cli.option_verbose()
def classify(model, input_file, output_file, train_fraction, type_count, seed, verbose):
    """Perform logistic regression
    """
    np.random.seed(seed)

    raw_data = io.read_file(input_file)
    shuffled_data = np.random.permutation(raw_data)
    split = int(shuffled_data.shape[0] * train_fraction)
    train_data, test_data = shuffled_data[:split], shuffled_data[split:]

    train_x, train_y = prepare_pair(train_data, type_count)
    test_x, test_y = prepare_pair(test_data, type_count)

    model = _select_model(model, verbose=verbose)
    model = model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    stats = _evaluate(prediction, test_y)

    io.vput(
        "{:>10}|{:>10}|{:>10}\n"
        "--------------------------------------\n"
        "{:>10}|{:>10,.2f}|{:>10,.2f}\n"
        "{:>10}|{:>10,.2f}|{:>10,.2f}\n",
        "",
        "0",
        "1",
        "0",
        stats[0, 0],
        stats[0, 1],
        "1",
        stats[1, 0],
        stats[1, 1],
        verbose=verbose,
    )

    raw_x, _ = prepare_pair(raw_data, type_count)
    probabilities = model.predict_proba(raw_x)[:, 1]
    coefs = build_coef_matrix(probabilities, raw_data.shape[0])
    io.save_npz(coefs, output_file)
