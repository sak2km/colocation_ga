"""CLI functions
"""
import logging

import click

from .utils import benchmark, cli, io, log_helper

LOG = logging.getLogger("M")


@click.command("analyze-error")
@cli.option_input_file()
@click.option("--room-count", default=50)
@click.option("--type-count", default=4)
def analyze_error(input_file, room_count, type_count):
    """Analyze error in room assignment
    """
    from .evaluate import error_analysis

    room_assignment = io.read_file(input_file)
    error_analysis.analyze_error(room_assignment, room_count, type_count)


@click.command(name="ga-assign-room")
@cli.option_config_file()
@click.option("-t", "--times", default=1, help="repeat for t number of times")
@cli.option_verbose()
@cli.option_benchmark()
@click.option("--shuffle-room", is_flag=True)
# Overwrite options from CLI
@click.option("--job-name", help="Job name")
@click.option("--seed", help="seed")
@click.option("--corr-matrix-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-path", type=click.Path(file_okay=False))
@click.option("--room-count", type=click.INT)
def ga_assign_room(config_file, times, verbose, benchmark_type, shuffle_room, **kwargs):
    """Assign room to sensors
    """
    import numpy as np
    import random
    import progressbar
    from .genetic_algorithm.tasks import strict_ga
    from .genetic_algorithm.data_loader import config_loader

    config = config_loader.load_config(config_file, verbose=verbose >= 2, **kwargs)
    io.make_dir(config.base_file_name)

    log_helper.configure_logger(
        verbose, output_file=config.join_name("colocation.log"), to_stderr=True
    )
    LOG.info("Started Task Strict GA")

    benchmark_manager = benchmark.BenchmarkManager.new(
        benchmark_type, output_path=config.base_file_name
    )
    benchmark_manager.start()

    if times == 1:
        _, accuracy, _ = strict_ga.run(config)
    else:
        accuracies = np.zeros(times)
        for i in progressbar.progressbar(range(times)):
            if shuffle_room and config.selected_rooms:
                random.shuffle(config.selected_rooms)

            config.seed = i
            _, accuracy, _ = strict_ga.run(config)
            accuracies[i] = accuracy

        io.save_npz(accuracies, config.join_name("accuracies.npz"))
        io.vput("max accuracy = {}", np.max(accuracies), verbose=verbose > 0)

    benchmark_manager.end()

    LOG.info("Ending Task")


@click.command("rank-learn")
@cli.option_input_file()
@cli.option_output_dir()
@click.option("--step", default=1)
@click.option("--seg-length", default=40)
@click.option("--n-epoch", default=1)
@click.option("--cuda", is_flag=True, default=False)
@cli.option_benchmark()
@cli.option_verbose()
def rank_learn(
    input_file, output_dir, step, seg_length, n_epoch, cuda, benchmark_type, verbose
):
    """run rank learn
    """
    output_path = io.make_dir(output_dir)
    log_helper.configure_logger(
        verbose, output_file=output_path.joinpath("rank-learn.log"), to_stderr=True
    )

    from .binary_classification import rank_learning

    if cuda:
        import torch

        LOG.info("Initialize CUDA...")
        torch.cuda.init()

    bm = benchmark.BenchmarkManager.new(benchmark_type)
    LOG.info("Start Rank Learning")
    bm.start()
    rank_learning.run_training(input_file, step, seg_length, n_epoch, cuda)
    bm.end()


@click.command("canny-edge")
@cli.option_input_file()
@cli.option_output_dir()
@cli.option_verbose()
@click.option("--sigma", default=1.0)
@click.option("--truncate", default=50.0)
@click.option("--savefig", is_flag=True, default=False)
@click.option("--use-scipy", is_flag=True, default=False)
def canny_edge(input_file, output_dir, verbose, sigma, truncate, savefig, use_scipy):
    """Canny Edge
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from .feature_extractor import canny_edge_detection
    from .utils import plot

    output_dir = io.make_dir(output_dir)
    log_helper.configure_logger(
        verbose, output_file=output_dir.joinpath("canny_edge.log"), to_stderr=True
    )

    data = io.read_file(input_file)

    for i in range(data.shape[0]):
        data[i] = data[i] - np.mean(data[i])
        data[i] = data[i] / np.linalg.norm(data[i])

    intermediate_steps = None

    edges = (
        canny_edge_detection.detect_edge_scipy(data, sigma)
        if use_scipy
        else canny_edge_detection.detect_edge(data, sigma, truncate, intermediate_steps)
    )

    out_filename = output_dir.joinpath("edges.npz")
    io.save_npz(edges, out_filename)

    LOG.debug("Saved detected edge in %s", str(out_filename))

    # if savefig:
    #     for i in range(50):
    #         LOG.debug("Generating diagram %d / %d", i + 1, 50)
    #         row_data = {
    #             "Type 1": edges[i * 4],
    #             "Type 2": edges[i * 4 + 1],
    #             "Type 3": edges[i * 4 + 2],
    #             "Type 4": edges[i * 4 + 3],
    #         }
    #         fig = plot.plot_each_series(row_data, f"Room {i}")
    #         fig.savefig(output_dir.joinpath(f"room_{i}.png"))
    #         plt.close(fig)


@click.command(name="pair-corr-coef")
@cli.option_input_file()
@cli.option_output_file()
@click.option("--absolute", is_flag=True, help="apply abs to the matrix")
@click.option("--rescale", is_flag=True, help="rescale between 1 and 0")
def pairwise_corrcoef(input_file, output_file, absolute, rescale):
    """Calculate pairwise correlation coefficient and save to a numpy file
    """
    import numpy as np

    data = io.read_npz(input_file)
    corr_matrix = np.corrcoef(data)
    if absolute:
        corr_matrix = np.abs(corr_matrix)
    if rescale:
        amax = np.max(corr_matrix)
        amin = np.min(corr_matrix)
        corr_matrix = (corr_matrix - amin) / (amax - amin)

    io.save_npz(corr_matrix, output_file)


@click.command(name="pair-dot")
@cli.option_input_file()
@cli.option_output_file()
@click.option("--absolute", is_flag=True, help="apply abs to the matrix")
@click.option("--rescale", is_flag=True, help="rescale between 1 and 0")
def pairwise_dot(input_file, output_file, absolute, rescale):
    """Calculate pairwise correlation coefficient and save to a numpy file
    """
    import numpy as np

    data = io.read_npz(input_file)
    corr_matrix = np.matmul(data, data.T)

    if absolute:
        corr_matrix = np.abs(corr_matrix)
    if rescale:
        amax = np.max(corr_matrix)
        amin = np.min(corr_matrix)
        corr_matrix = (corr_matrix - amin) / (amax - amin)

    io.save_npz(corr_matrix, output_file)


@click.command()
@cli.option_input_file()
@cli.option_output_file()
@click.option("--grid-line", default=None)
@click.option("--show-room-num", is_flag=True)
def heatmap(input_file, output_file, grid_line, show_room_num):
    """Show a heatmap of a 2D matrix
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn
    import numpy as np

    data = io.read_npz(input_file)
    np.fill_diagonal(data, 0)
    fig = plt.figure(figsize=(12, 10))
    ax = seaborn.heatmap(data, square=True)

    if grid_line:
        size = len(data)
        ticks = (
            list(range(0, size, 4))
            if (grid_line.isdigit())
            else io.read_file(grid_line)
        )
        plt.hlines(ticks, 0, size, colors="cyan", linewidths=0.1)
        plt.vlines(ticks, 0, size, colors="cyan", linewidths=0.1)

    if show_room_num:
        from .data_processing.keti_loader import room_ids

        ax2 = ax.twinx()
        ax2.set_ylim([51, -1])
        ax2.set_yticks(np.arange(49, -1, -1) + 0.5)
        ax2.set_yticklabels(list(reversed(room_ids())))

    io.touch(output_file)
    fig.savefig(output_file, dpi=192)


@click.command("gaussian")
@cli.option_input_file()
@cli.option_output_file()
@click.option("--sigma", type=click.INT)
def gaussian_filter(input_file, output_file, sigma):
    """Smooth by a gaussian filter
    """
    from .feature_extractor.canny_edge_detection import gaussian_filter1d

    data = io.read_file(input_file).astype(float)
    smoothed = gaussian_filter1d(data, sigma=sigma, axis=1)
    assert smoothed.shape == data.shape
    io.save_npz(smoothed, output_file)


@click.command("ahu-vav-load")
@cli.option_input_dir()
@cli.option_output_dir()
def ahu_vav_load(input_dir, output_dir):
    from .data_processing import ahu_vav_loader

    data, v_counts = ahu_vav_loader.concatenate_data(input_dir)
    output_dir = io.make_dir(output_dir)

    io.save_npz(data, output_dir.joinpath("data.npz"))
    io.save_npz(v_counts, output_dir.joinpath("v_counts.npz"))


@click.command("ahu-vav-dezhi-load")
@cli.option_input_file()
@cli.option_output_dir()
@click.option("--n-ahu", default=8)
def ahu_vav_dezhi_load(input_file, output_dir, n_ahu):
    """Load dezhi's event for ahu_vav
    """
    import numpy as np
    from .data_processing.ahu_vav_loader import read_dezhi_csv

    data = io.read_file(input_file)
    data, v_counts = read_dezhi_csv(data, n_ahu)

    assert data.shape == (121, 2688), "Shape Mismatch: %s" % str(data.shape)
    assert len(v_counts) == 8
    assert np.sum(v_counts) == 121 - 8

    output_dir = io.make_dir(output_dir)
    io.save_npz(data, output_dir.joinpath("data.npz"))
    io.save_npz(v_counts, output_dir.joinpath("v_counts.npz"))


@click.command("ga-ahu")
@cli.option_config_file()
@click.option("--vav-counts", required=True)
@click.option("--corr-matrix-path", required=True)
@cli.option_verbose()
def ga_ahu(config_file, vav_counts, corr_matrix_path, verbose):
    from .genetic_algorithm.tasks import ahu_iterative_ga
    import numpy as np

    config = io.read_json(config_file)
    log_helper.configure_logger(verbose, to_stderr=True)
    v_counts = io.read_file(vav_counts)
    n_vav = np.sum(v_counts)
    n_population = config["n_population"]
    corr_matrix = io.read_file(corr_matrix_path)

    perfect_corr = ahu_iterative_ga.perfect_corr_matrix(v_counts)
    accuracy_func = ahu_iterative_ga.compile_vav_ahu_score(perfect_corr, v_counts)
    # modify here:
    score_func = ahu_iterative_ga.compile_vav_ahu_score(corr_matrix, v_counts)
    result = ahu_iterative_ga.run_ga(
        initial_population=np.array(
            [np.random.permutation(n_vav) for _ in range(n_population)]
        ).reshape(n_population, -1, 1),
        score_func = score_func,
        **config
    )
    accuracy = accuracy_func(result)
    print("Accuracy = " + str(accuracy))
   

@click.command("ahu-mask-refine")
@cli.option_config_file()
@cli.option_verbose()
def ahu_mask_refine(config_file, verbose):
    config = io.read_json(config_file)
    log_helper.configure_logger(verbose, to_stderr=True)

    v_counts = io.read_file(config["v_counts_path"])
    ts_data = io.read_file(config["ts_data_path"])
    ts_ahu, ts_vav = ts_data[:len(v_counts)], ts_data[len(v_counts):]

    from .genetic_algorithm.tasks import ahu_iterative_ga

    ahu_iterative_ga.iterative_refine(
        ts_vav=ts_vav,
        ts_ahu=ts_ahu,
        v_counts=v_counts,
        **config
    )



@click.command("visualize")
@click.argument(
    "graph-type", type=click.Choice(["accuracy", "room-signal", "compare-ts"])
)
@cli.option_input_file()
@cli.option_output_file()
@click.option("--title")
@click.option("--type-count", default=4)
@cli.option_verbose()
def visualize(graph_type, input_file, output_file, title, type_count, verbose):
    """plot a graph
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    from .utils import plot

    log_helper.configure_logger(verbose, to_stderr=True)

    if not output_file:
        dir_path = Path(input_file).parent
        output_file = str(dir_path.joinpath(graph_type + ".png"))
    else:
        output_file = io.touch(output_file)

    if graph_type == "accuracy":
        data = io.read_file(input_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        for row in data:
            plt.plot(row)
        plt.ylabel("accuracy")
        plt.xlabel("iterations")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(str(output_file))
    elif graph_type == "room-signal":
        data = io.read_file(input_file)
        out_dir = output_file.parent
        out_name = output_file.stem
        room_count = data.shape[0] // type_count
        for i in range(room_count):
            LOG.debug("Generating diagram %d / %d", i + 1, room_count)
            row_data = {f"Type {t}": data[i * type_count + t] for t in range(type_count)}
            fig = plot.plot_each_series(row_data, f"Room {i}")
            fig.savefig(out_dir.joinpath(f"{out_name}_{i}.png"))
            plt.close(fig)


@click.group("Colocation-CLI")
def colocation_cli():
    """Colocation CLI Interface
    """
    pass


colocation_cli.add_command(analyze_error)
colocation_cli.add_command(ga_assign_room)
colocation_cli.add_command(rank_learn)
colocation_cli.add_command(canny_edge)
colocation_cli.add_command(gaussian_filter)
colocation_cli.add_command(pairwise_corrcoef)
colocation_cli.add_command(pairwise_dot)
colocation_cli.add_command(heatmap)
colocation_cli.add_command(ahu_vav_load)
colocation_cli.add_command(ahu_vav_dezhi_load)
colocation_cli.add_command(ga_ahu)
colocation_cli.add_command(ahu_mask_refine)
colocation_cli.add_command(visualize)
