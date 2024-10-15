"""
Plot the training data - to e.g. visualize the transforms

"""

import argparse


def main(*, step: int, epochs: int):
    """
    Read the DICOMs from disk () Create the data config

    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the training data")
    parser.add_argument(
        "--step",
        type=int,
        help="Interval between plots - step of 1 plots all data",
        default=1,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="How many complete passes over the training data to make",
        default=1,
    )

    main(**vars(parser.parse_args()))
