"""Generate data using feature vectors
"""
import click

from . import data_feature_extractor
from ..utils import io


@click.command()
@click.option(
    "-p",
    "--raw-data-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the raw data file",
    required=True,
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=False, file_okay=False),
    help="Path to the directory for output",
    required=True,
)
def extract_feature(raw_data_path: str = "", output_dir: str = ""):
    """Run feature extractor
    """
    x = io.read_npz(raw_data_path)
    feature_extractor = data_feature_extractor.data_feature_extractor(x)

    output_dir = io.make_dir(output_dir)

    feats = feature_extractor.getF_1994_Li()
    print("Li, 1994: {}".format(feats.shape))
    io.save_npz(feats, output_dir.joinpath("Li_1994.npz"))

    # feats = feature_extractor.getF_2012_Calbimonte()
    # print("Calbimonte, 2012: {}".format(feats.shape))
    # io.save_npz(feats, output_dir.joinpath("Calbimonte_2012.npz"))

    feats = feature_extractor.getF_2015_Gao()
    print("Gao, 2015: {}".format(feats.shape))
    io.save_npz(feats, output_dir.joinpath("Gao_2015.npz"))

    feats = feature_extractor.getF_2015_Hong()
    # feats = data_feature_extractor.window_feature(
    #     x, data_feature_extractor.get_statF_on_window, 4, overlapping=2
    # ).reshape((x.shape[0], -1))
    print("Hong, 2015: {}".format(feats.shape))
    io.save_npz(feats, output_dir.joinpath("Hong_2015.npz"))

    feats = feature_extractor.getF_2015_Bhattacharya()
    print("Bhattacharya, 2015: {}".format(feats.shape))
    io.save_npz(feats, output_dir.joinpath("Bhattacharya_2015.npz"))

    feats = feature_extractor.getF_2015_Balaji()
    print("Balaji, 2015: {}".format(feats.shape))

    io.save_npz(feats, output_dir.joinpath("Balaji_2015.npz"))

    feats = feature_extractor.getF_2016_Koh()
    print("Koh, 2016: {}".format(feats.shape))
    io.save_npz(feats, output_dir.joinpath("Koh_2016.npz"))


def main():
    """Main entrance
    """
    extract_feature()


if __name__ == "__main__":
    main()
