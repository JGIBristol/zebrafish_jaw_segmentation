"""
Plot the training data - to e.g. visualize the transforms

"""

import pathlib
import argparse

import torch
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt

from fishjaw.util import util
from fishjaw.model import data
from fishjaw.visualisation import images_3d


def main(*, step: int, epochs: int):
    """
    Read the DICOMs from disk () Create the data config

    """
    # Create training config
    config = util.userconf()
    torch.manual_seed(config["torch_seed"])
    rng = np.random.default_rng(seed=config["test_train_seed"])

    train_subjects, val_subjects, _ = data.read_dicoms_from_disk(config, rng)
    data_config = data.DataConfig(config, train_subjects, val_subjects)

    output_dir = (
        pathlib.Path(__file__).parents[1]
        / util.config()["script_output"]
        / "train_data"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Epochs
    for epoch in range(0, epochs, step):
        # Batches
        for i, batch in enumerate(data_config.train_data):
            images = batch[tio.IMAGE][tio.DATA]
            masks = batch[tio.LABEL][tio.DATA]
            # Images per batch
            for j, (image, mask) in enumerate(zip(images, masks)):
                out_path = str(
                    output_dir / f"traindata_epoch_{epoch}_batch_{i}_img_{j}.png"
                )

                fig, _ = images_3d.plot_slices(
                    image.squeeze().numpy(), mask.squeeze().numpy()
                )
                fig.savefig(out_path)
                plt.close(fig)


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
