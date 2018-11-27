"""Logging utils
"""

import logging


def configure_logger(verbose, output_file=None, to_stderr=False):
    """Configure the main logger
    """
    logger = logging.getLogger("M")

    logger.setLevel(
        logging.ERROR
        if verbose == 0
        else logging.WARNING
        if verbose == 1
        else logging.INFO
        if verbose == 2
        else logging.DEBUG
    )

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    if output_file:
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if to_stderr:
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
