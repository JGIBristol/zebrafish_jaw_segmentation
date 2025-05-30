"""
Fine tune a model on a small dataset of quadrates,

"""

import argparse

from fishjaw.util import util
from fishjaw.transfer import data


def train(*, epochs: int, **kwargs):
    """
    Read in the training data and use it to train a model.
    Make some plots of the loss, the inference on the testing data and output some metrics
    """
    config = util.userconf()

    train_subjects, val_subjects, test_subjects = data.quadrate_data(config)


def fine_tune(
    *, base_model: str, epochs: int, train_layers: list[int], train_all: bool, **kwargs
):
    """
    Read in the training data, load in the base model and fine tune it
    Make some plots of the loss, the inference on the testing data and output some metrics
    """
    config = util.userconf()

    train_subjects, val_subjects, test_subjects = data.quadrate_data(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Either train or fine tune a model on a small dataset of quadrates"
    )

    # Separate CLI args for training and fine tuning
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser(
        "train", help="Train a model on a small dataset of quadrates"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=650,
        help="Number of epochs of the quadrate data to use",
    )
    train_parser.set_defaults(func=train)

    fine_tune_parser = subparsers.add_parser(
        "fine_tune",
        help="Fine tune a model on a small dataset of quadrates",
    )
    fine_tune_parser.add_argument("base_model", help="Base model to fine-tune")
    fine_tune_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    group = fine_tune_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train-layers",
        type=int,
        nargs="+",
        help="Layers to train, e.g. 0 1 2",
        choices=list(range(6)),
    )
    group.add_argument("--train-all", action="store_true", help="Train all layers")
    fine_tune_parser.set_defaults(func=fine_tune)

    args = parser.parse_args()
    args.func(**vars(args))
