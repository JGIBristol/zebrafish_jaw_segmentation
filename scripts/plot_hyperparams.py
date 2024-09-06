"""
Plot the results from the hyperparam search

"""

import pathlib
import argparse

import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def _plot_coarse():
    """
    Read the losses and indices from the coarse search,
    then plot them colour coded by LR

    """
    paths = sorted(
        list((pathlib.Path(__file__).parents[1] / "tuning_output" / "coarse").glob("*"))
    )

    fig, axes = plt.subplots(1, 2, sharey=True)
    cmap = plt.get_cmap("viridis")

    for path in tqdm(paths):
        # Get the training and validation loss from each
        # Average over batches
        train_loss = np.load(path / "train_losses.npy").mean(axis=1)
        val_loss = np.load(path / "val_losses.npy").mean(axis=1)

        # Get the LR
        with open(path / "config.yaml") as f:
            params = yaml.safe_load(f)
        lr = params["learning_rate"]

        # Scale to between 0 and 1
        scaled_lr = (np.log10(lr) + 6) / 7

        # Get the colour from the LR
        colour = cmap(scaled_lr)

        axes[0].plot(train_loss, color=colour)
        axes[1].plot(val_loss, color=colour)

    # Add a colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-6, vmax=1)), ax=axes
    )
    cbar.set_label("Learning rate")

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")

    for axis in axes:
        axis.set_xticks([])

    fig.savefig("coarse_search.png")


def main(mode: str):
    if mode == "coarse":
        _plot_coarse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the results of hyperparameter search"
    )

    parser.add_argument(
        "mode",
        type=str,
        choices={"coarse", "med", "fine"},
        help="Granularity of the search.",
    )

    main(**vars(parser.parse_args()))
