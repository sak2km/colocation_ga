"""Commonly used CLI options
"""

import click


def option_input_file():
    """Get a click option for input_file
    """
    return click.option(
        "-f",
        "--input-file",
        type=click.Path(exists=True, dir_okay=False),
        help="Input file path",
    )


def option_output_file():
    """Get a click option for output_file
    """
    return click.option(
        "-o", "--output-file", type=click.Path(dir_okay=False), help="output file name"
    )


def option_input_dir():
    """Get a click option for input_dir
    """
    return click.option(
        "-d",
        "--input-dir",
        type=click.Path(exists=True, file_okay=False),
        help="Input file directory",
    )


def option_output_dir():
    """Get a click option for output_dir
    """
    return click.option(
        "-o", "--output-dir", type=click.Path(file_okay=False), help="output directory"
    )


def option_config_file():
    """Get a click option for config_file
    """
    return click.option(
        "-c",
        "--config-file",
        type=click.Path(dir_okay=False),
        help="configuration file path",
    )


def option_verbose():
    """Get a verbose option
    """
    return click.option("-v", "--verbose", count=True, help="verbose level")


def option_benchmark():
    """Get a benchmark option
    """
    return click.option(
        "--benchmark-type",
        default="None",
        type=click.Choice(["None", "TimeIt", "cProfile"]),
    )
