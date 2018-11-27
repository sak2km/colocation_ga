"""Commands for evaluation
"""

import click
import numpy as np
from scipy import stats
import terminaltables

from ..utils import cli, io, functional


@click.command("corr-matrix")
@cli.option_input_file()
@cli.option_output_file()
@click.option("-n", "--num-types", default=4, help="Number of types of sensor")
@cli.option_verbose()
def eval_corr_matrix(input_file, output_file, num_types, verbose):
    """Evaluate a correlation matrix
    """
    data = io.read_npz(input_file)

    sensor_count = data.shape[0]

    in_room = []
    out_room = []

    for i in range(sensor_count):
        for j in range(sensor_count):
            if i == j:
                continue
            if i % num_types == j % num_types:
                in_room.append(data[i, j])
            else:
                out_room.append(data[i, j])

    in_summary = stats.describe(np.array(in_room))
    out_summary = stats.describe(np.array(out_room))

    table = [
        [in_summary.mean, *(in_summary.minmax), in_summary.variance],
        [out_summary.mean, *(out_summary.minmax), out_summary.variance],
    ]

    str_table = terminaltables.AsciiTable(table).table

    io.vput(str_table, verbose=verbose >= 1)
    io.save_txt(str_table, output_file)


@click.group("evaluate")
def eval_commands():
    """Evaluate some thing
    """
    pass


eval_commands.add_command(eval_corr_matrix)
