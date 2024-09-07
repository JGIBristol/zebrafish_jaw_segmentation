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

    n_filter_fig, n_filter_ax = plt.subplots(1, 2, sharey=True)

    for path in tqdm(paths):
        # Get the training and validation loss from each
        # Average over batches
        train_loss = np.load(path / "train_losses.npy").mean(axis=1)
        val_loss = np.load(path / "val_losses.npy").mean(axis=1)

        # Get the LR and n filters
        with open(path / "config.yaml") as f:
            params = yaml.safe_load(f)
        lr = params["learning_rate"]
        n_filters = params["model_params"]["n_initial_filters"]

        # Scale to between 0 and 1
        scaled_lr = (np.log10(lr) + 6) / 7
        scaled_filters = (n_filters - 4) / 20

        # Get the colour from the LR
        colour = cmap(scaled_lr)
        axes[0].plot(train_loss, color=colour)
        axes[1].plot(val_loss, color=colour)

        # Get the colour from the n filters
        colour = cmap(scaled_filters)
        n_filter_ax[0].plot(train_loss, color=colour)
        n_filter_ax[1].plot(val_loss, color=colour)


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
    plt.close(fig)
    
    cbar = n_filter_fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=4, vmax=24)), ax=n_filter_ax
    )
    cbar.set_label("Number of initial filters")
    n_filter_ax[0].set_title("Training loss")
    n_filter_ax[1].set_title("Validation loss")
    for axis in n_filter_ax:
        axis.set_xticks([])
    n_filter_fig.savefig("coarse_search_n_filters.png")
    plt.close(n_filter_fig)



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
