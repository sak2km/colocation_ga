"""STL Decomposition
"""

import functools
import multiprocessing as mp

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import stldecompose as stl

from ..utils import cli, io


def _decompose_vector(vector, period):
    data = pd.DataFrame(vector.reshape(-1, 1))
    data.index = pd.DatetimeIndex(np.arange(vector.size))
    decomposed = stl.decompose(df=data, period=period)
    return decomposed


@click.command("STL")
@click.argument("ids", nargs=-1, type=click.INT)
@cli.option_input_file()
@cli.option_output_dir()
@click.option("--period", default=7)
@click.option("--prefix", default="")
@click.option("--draw", is_flag=True, default=False)
@click.option("--type-id", default=-1)
@click.option("--type-count", default=-1)
@cli.option_verbose()
def decompose(
    ids, input_file, output_dir, period, verbose, prefix, draw, type_id, type_count
):
    """Decompose
    """
    # pylint: disable=no-member
    io.vput(f"prefix={prefix}, period={period}", verbose=verbose >= 1)
    raw_data = io.read_file(input_file)
    output_dir = io.make_dir(output_dir)

    seasonals = np.zeros_like(raw_data)
    trends = np.zeros_like(raw_data)
    resids = np.copy(raw_data)

    if not ids:
        ids = (
            list(range(raw_data.shape[0]))
            if type_id == -1
            else list(range(type_id, raw_data.shape[0], type_count))
        )

    for i in progressbar.progressbar(ids):
        decomposed = _decompose_vector(raw_data[i], period)
        seasonals[i] = decomposed.seasonal.values.reshape(-1)
        trends[i] = decomposed.trend.values.reshape(-1)
        resids[i] = decomposed.resid.values.reshape(-1)

        if draw:
            fig = plt.figure(figsize=(6, 6))
            plt.subplot("411")
            plt.plot(raw_data[i])
            plt.title(f"Room {i}, Period {period}")
            plt.subplot("412")
            plt.plot(decomposed.seasonal.values)
            plt.subplot("413")
            plt.plot(decomposed.trend.values)
            plt.subplot("414")
            plt.plot(decomposed.resid.values)

            plt.savefig(output_dir.joinpath(f"{prefix}{i}.png"), dpi=300)
            fig.close()  # Avoid memory leak

    io.save_npz(seasonals, output_dir.joinpath("{prefix}seasonals.npz"))
    io.save_npz(trends, output_dir.joinpath("{prefix}trends.npz"))
    io.save_npz(resids, output_dir.joinpath("{prefix}resids.npz"))


def _type_std_seasonal(dataframe, type_count, length):
    # pylint: disable=no-member
    seasonal_vals = np.zeros((len(dataframe.columns), len(dataframe)))
    for i in range(len(dataframe.columns)):
        seasonal_vals[i] = stl.decompose(dataframe[i], period=length).seasonal.values
    type_stds = [
        np.mean(np.std(seasonal_vals[t::type_count], axis=0)) for t in range(type_count)
    ]
    return length, np.mean(type_stds)


def _type_std(dataframe, type_count, type_id, length):
    seasonal_vals = np.zeros((len(dataframe.columns) // type_count, len(dataframe)))
    for i in range(type_id, len(dataframe.columns), type_count):
        seasonal_vals[i // type_count] = stl.decompose(
            dataframe[i], period=length
        ).seasonal.values
    type_std = np.mean(np.std(seasonal_vals, axis=0))
    return length, type_std


@click.command("STL-optimize")
@click.option("--type-count", default=4)
@click.option("--min-season", default=5, help="minimum season length")
@click.option("--max-season", default=10000, help="maximum season length")
@click.option("--step", default=1)
@click.option("--n-process", default=8)
@cli.option_input_file()
@cli.option_output_dir()
@cli.option_verbose()
def stl_optimize(
    type_count, min_season, max_season, step, n_process, input_file, output_dir, verbose
):
    """Calculate the season length with minimum std. of seasonal pattern within a type.
    """
    raw_data = io.read_file(input_file)
    dataframe = pd.DataFrame(raw_data.T)
    dataframe.index = pd.DatetimeIndex(np.arange(len(dataframe)))
    lengths = np.arange(min_season, max_season + 1, step)

    pool = mp.Pool(n_process)
    table = np.zeros((len(lengths), 2))

    get_std = functools.partial(_type_std_seasonal, dataframe, type_count)
    for i, (length, std) in zip(
        progressbar.progressbar(range(len(table))),
        pool.imap_unordered(get_std, lengths, 1),
    ):
        table[i, 0] = length
        table[i, 1] = std

    output_dir = io.make_dir(output_dir)
    io.save_npz(table, output_dir.joinpath("stds.npz"))

    io.vput("{!s}", table, verbose=verbose >= 3)

    min_len, min_std = table[np.argmin(table[:, 1])]
    io.vput("Best: length={}, std={}", min_len, min_std, verbose=verbose)


@click.command("STL-optimize-per-type")
@click.option("--type-count", default=4)
@click.option("--min-season", default=5, help="minimum season length")
@click.option("--max-season", default=10000, help="maximum season length")
@click.option("--step", default=1)
@click.option("--n-process", default=8)
@cli.option_input_file()
@cli.option_output_dir()
@cli.option_verbose()
def stl_optimize_per_type(
    type_count, min_season, max_season, step, n_process, input_file, output_dir, verbose
):
    """Optimize the length of periods per type
    """
    raw_data = io.read_file(input_file)
    dataframe = pd.DataFrame(raw_data.T)
    dataframe.index = pd.DatetimeIndex(np.arange(len(dataframe)))
    lengths = np.arange(min_season, max_season + 1, step)

    pool = mp.Pool(n_process)
    output_dir = io.make_dir(output_dir)

    for t in range(type_count):
        io.vput("Type [{}]", t, verbose=verbose)
        table = np.zeros((len(lengths), 2))
        get_std = functools.partial(_type_std, dataframe, type_count, t)
        
        if n_process > 1:
            for i, (length, std) in zip(
                progressbar.progressbar(range(len(table))),
                pool.imap_unordered(get_std, lengths, 1),
            ):
                table[i, 0] = length
                table[i, 1] = std
        else:
            for i, (length, std) in zip(
                progressbar.progressbar(range(len(table))),
                map(get_std, lengths),
            ):
                table[i, 0] = length
                table[i, 1] = std
        
        io.vput("", verbose=verbose)
        io.vput("{!s}", table, verbose=verbose >= 3)
        io.save_npz(table, output_dir.joinpath(f"stds_{t}.npz"))

        min_len, min_std = table[np.argmin(table[:, 1])]
        io.vput("Best: length={}, std={}", min_len, min_std, verbose=verbose)
