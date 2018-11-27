"""Command line interface
"""

import random

import click
import numpy as np
import scipy.stats
import progressbar
import rapidjson

from . import tasks
from ..utils import io
from ..utils import cli
from .data_loader import config_loader


@click.group()
def _cli():
    pass


@click.command(name="gen-config")
@cli.option_config_file()
def gen_config(config_file):
    """Generate a configuration file
    """
    config = config_loader.ColocationConfig()
    io.save_json(config.to_dict(), config_file)


@click.command(name="assign-room")
@cli.option_config_file()
@click.option("-t", "--times", default=1, help="repeat for t number of times")
@cli.option_verbose()
@click.option("--job-name", help="Job name")
@click.option("--seed", help="seed")
@click.option("--corr-matrix-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-path", type=click.Path(file_okay=False))
@click.option("--room-count", type=click.INT)
@click.option("--shuffle-room", is_flag=True)
def assign_room(config_file, times, verbose, shuffle_room, **kwargs):
    """Assign room to sensors to optimize the sum of intra-room correlation coefficients
    """
    with open(config_file) as file:
        config_dict = rapidjson.load(file)
    config_dict["verbose"] = verbose >= 2

    for key, val in kwargs.items():
        if val:
            config_dict[key] = val

    config = config_loader.ColocationConfig(**config_dict)
    io.make_dir(config.base_file_name)

    if times == 1:
        tasks.TASKS[config.task].run(config)
        return

    accuracies = np.zeros(times)
    for i in progressbar.progressbar(range(times)):
        if shuffle_room:
            random.shuffle(config.selected_rooms)
        config.seed = i
        _, accu, _ = tasks.TASKS[config.task].run(config)
        accuracies[i] = accu

    io.save_npz(accuracies, config.join_name("accuracies.npz"))
    summary = scipy.stats.describe(accuracies)
    if verbose:
        print("mean accuracy: {}, minmax: {}".format(summary.mean, summary.minmax))


if __name__ == "__main__":
    _cli.add_command(gen_config)
    _cli.add_command(assign_room)
    _cli()
