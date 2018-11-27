"""Visualize some data structures
"""
import pathlib

import click
import seaborn
import matplotlib.pyplot as plt

from ..utils import cli, io


@click.command()
@cli.option_input_file()
@cli.option_output_file()
@click.option("--show", count=True)
@cli.option_verbose()
def heatmap(input_file, output_file, show, verbose):
    """Show a heatmap of a 2D matrix
    """
    io.vput(
        "InputFile: {}, OutputFile: {}, Show: {}",
        input_file,
        output_file,
        show,
        verbose=verbose >= 3,
    )

    data = io.read_npz(input_file)
    fig = seaborn.heatmap(data).get_figure()
    io.touch(output_file)
    fig.savefig(output_file, dpi=192)

    if show:
        fig.show()


@click.command()
@click.argument("input-files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@cli.option_output_file()
@click.option("-t", "--title")
@click.option("-l", "--labels", multiple=True)
def plot_accuracy(input_files, output_file, title, labels):
    """Simple plot
    """
    if not labels:
        labels = [pathlib.Path(path).name for path in input_files]

    assert len(labels) == len(
        input_files
    ), "Number of labels must match number of files"

    for input_file in input_files:
        data = io.read_file(input_file)
        plt.plot(data)

    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracies")
    plt.legend(labels)

    io.touch(output_file)
    plt.savefig(output_file, dpi=192)


@click.group("visualize")
def visualize_cli():
    """Visualize data saved by the program"""
    pass


visualize_cli.add_command(heatmap)
visualize_cli.add_command(plot_accuracy)
