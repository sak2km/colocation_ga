"""Main entrance of this module
"""

# import click

# from . import evaluate
# from . import visualizer
# from . import feature_extractor
# from . import binary_classification
# from .binary_classification import logistic_regression
from . import cli


# @click.group()
# def _cli():
#     pass


# _cli.add_command(evaluate.eval_commands)
# _cli.add_command(visualizer.visualize_cli)
# # _cli.add_command(feature_extractor.test_shapelet)
# _cli.add_command(binary_classification.classify)
# _cli.add_command(feature_extractor.stl)
# _cli.add_command(feature_extractor.stl_decomposition.stl_optimize)
# _cli.add_command(feature_extractor.stl_decomposition.stl_optimize_per_type)
# _cli.add_command(logistic_regression.lr_outer)
# _cli.add_command(cli.analyze_error)
# _cli.add_command(cli.ga_assign_room)


if __name__ == "__main__":
    cli.colocation_cli()
