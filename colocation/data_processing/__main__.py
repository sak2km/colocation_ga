"""Main
"""

import click
import numpy as np
from scipy import fftpack, signal
from sklearn import preprocessing

from . import keti_loader, time_series
from ..utils import cli, io, functional, progress


@click.group()
def _cli():
    pass


@click.command(name="trim-align")
@cli.option_input_dir()
@cli.option_output_file()
@click.option("-i", "--interval", default=5, help="interval for aligned data")
@cli.option_verbose()
def trim_and_align(input_dir, output_file, interval, verbose):
    """load raw data, trim and align the data, and save the resulting file
    """
    io.vput("Loading data to memory...", verbose=verbose)
    data_list = keti_loader.load(input_dir, show_progress=True)

    io.vput("\nTrim and align data...", verbose=verbose)
    aligned_data = time_series.trim_and_align(data_list, interval)
    io.vput("The shape of the trimmed data is {}", aligned_data.shape, verbose=verbose)

    io.vput("\nSave the file as {}", output_file, verbose=verbose)
    io.save_npz(aligned_data, output_file)


@click.command(name="autocorrelation")
@cli.option_input_file()
@cli.option_output_file()
@cli.option_verbose()
def auto_correlation(input_file, output_file, verbose):
    """Load raw data, and save the autocorrelations
    """

    data = io.read_npz(input_file)
    nrow, ncol = data.shape

    updater = progress.get_update_function(nrow, noop=(not verbose))
    auto_corr = functional.array_map(
        time_series.autocorr, data, out_shape=(nrow, ncol), update_callback=updater
    )

    io.save_npz(auto_corr, output_file)


@click.command(name="pair-corrcoef")
@cli.option_input_file()
@cli.option_output_file()
@click.option(
    "--absolute", count=True, help="If this option exists, then abs is applied."
)
@click.option("--rescale", count=True, help="rescale between 1 and 0")
def pairwise_corrcoef(input_file, output_file, absolute, rescale):
    """Calculate pairwise correlation coefficient and save to a numpy file
    """
    data = io.read_npz(input_file)
    corr_matrix = np.corrcoef(data)
    corr_matrix = np.abs(corr_matrix) if absolute else corr_matrix
    corr_matrix = time_series.scale(corr_matrix, 1, 0) if rescale else corr_matrix

    io.save_npz(corr_matrix, output_file)


@click.command(name="normalize")
@cli.option_input_file()
@cli.option_output_file()
@click.option("-n", "--num-types", default=4, help="number of types.")
def normalize_by_type(input_file, output_file, num_types):
    """Normalize (standard core) data by type
    """
    if num_types < 1:
        print("num_types must be positive!")
        exit(-1)

    data = io.read_npz(input_file)

    for start in range(num_types):
        data[start::num_types] = time_series.normalize(data[start::num_types])

    io.save_npz(data, output_file)


@click.command(name="fft")
@cli.option_input_file()
@cli.option_output_file()
@click.option("-l", "--length", default=None, type=click.INT, help="Length of FFT")
def fft(input_file, output_file, length):
    """Extract FFT coefficients from time series
    """
    data = io.read_npz(input_file)
    coef = fftpack.rfft(data, axis=1, n=length, overwrite_x=True)
    io.save_npz(coef, output_file)


@click.command(name="standardize")
@cli.option_input_file()
@cli.option_output_file()
def standardize(input_file, output_file):
    """Standardize each row
    """
    data = io.read_npz(input_file)
    data = preprocessing.scale(data, axis=1)
    io.save_npz(data, output_file)


@click.command(name="smooth")
@cli.option_input_file()
@cli.option_output_file()
@click.option("--window", default=101)
def smooth(input_file, output_file, window):
    """Smoothe the curve
    """
    data = io.read_npz(input_file)
    data = signal.savgol_filter(data, window, 3, axis=1)
    io.save_npz(data, output_file)


@click.command(name="reduce-bg-noise")
@cli.option_input_file()
@cli.option_output_file()
@click.option("--threshold", default=1.0)
@click.option("--threshold-type", default="std", type=click.Choice(["std"]))
def reduce_bg_noise(input_file, output_file, threshold, threshold_type):
    """Remove background noise
    """
    dataset = io.read_file(input_file)
    dataset = time_series.suppress_noise(dataset, threshold)
    io.save_npz(dataset, output_file)


if __name__ == "__main__":
    _cli.add_command(trim_and_align)
    _cli.add_command(auto_correlation)
    _cli.add_command(pairwise_corrcoef)
    _cli.add_command(normalize_by_type)
    _cli.add_command(fft)
    _cli.add_command(standardize)
    _cli.add_command(smooth)
    _cli.add_command(reduce_bg_noise)
    _cli()
