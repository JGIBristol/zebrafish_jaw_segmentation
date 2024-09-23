"""
Plot the results from the hyperparam search

"""

import pathlib
import argparse
from dataclasses import dataclass, fields

import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from fishjaw.images import metrics


@dataclass
class RunInfo:
    score: float
    lr: float
    n_filters: int
    batch_size: int
    alpha: float
    epochs: int
    one_minus_lambda: float


def _batch_plot(paths: list[pathlib.Path]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, sharey=True)
    cmap = plt.get_cmap("viridis")

    for path in tqdm(paths):
        try:
            train_loss = np.load(path / "train_losses.npy").mean(axis=1)
            val_loss = np.load(path / "val_losses.npy").mean(axis=1)
        except FileNotFoundError:
            continue

        # Get the batch size
        with open(path / "config.yaml") as f:
            params = yaml.safe_load(f)
        batch_size = params["batch_size"]

        # Scale to between 0 and 1
        scaled = (batch_size - 2) / 12

        # Get the colour from the LR
        colour = cmap(scaled)
        axes[0].plot(train_loss, color=colour)
        axes[1].plot(val_loss, color=colour)

    # Add a colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=2, vmax=12)), ax=axes
    )
    cbar.set_label("Batch size")
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xticks([])

    return fig


def _lr_plot(paths: list[pathlib.Path]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, sharey=True)
    cmap = plt.get_cmap("viridis")

    for path in tqdm(paths):
        try:
            train_loss = np.load(path / "train_losses.npy").mean(axis=1)
            val_loss = np.load(path / "val_losses.npy").mean(axis=1)
        except FileNotFoundError:
            continue

        # Get the LR and n filters
        with open(path / "config.yaml") as f:
            params = yaml.safe_load(f)
        lr = params["learning_rate"]
        n_filters = params["model_params"]["n_initial_filters"]
        batch_size = params["batch_size"]

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

    return fig


def _filter_plot(paths: list[pathlib.Path]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, sharey=True)
    cmap = plt.get_cmap("viridis")

    for path in tqdm(paths):
        # Get the training and validation loss from each
        # Average over batches
        try:
            train_loss = np.load(path / "train_losses.npy").mean(axis=1)
            val_loss = np.load(path / "val_losses.npy").mean(axis=1)
        except FileNotFoundError:
            continue

        # Get the LR and n filters
        with open(path / "config.yaml") as f:
            params = yaml.safe_load(f)
        n_filters = params["model_params"]["n_initial_filters"]

        # Scale to between 0 and 1
        scaled_filters = (n_filters - 4) / 20

        # Get the colour from the n filters
        colour = cmap(scaled_filters)
        axes[0].plot(train_loss, color=colour)
        axes[1].plot(val_loss, color=colour)

    # Add a colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=4, vmax=24)), ax=axes
    )
    cbar.set_label("N filters")
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xticks([])

    return fig


def _plot_scatters(data_dir: pathlib.Path, metric: str) -> plt.Figure:
    """
    Plot scatter plots and a histogram of dice scores if they exist
    metric must either be "dice" or "loss"

    """
    data_dirs = list(data_dir.glob("*"))
    runs = []
    for dir_ in data_dirs:
        # We might still be running, in which case the last dir will be incomplete
        if metric == "dice":
            try:
                score = _dicescore(dir_)
            except FileNotFoundError:
                continue
        elif metric == "loss":
            try:
                score = 1 - np.load(dir_ / "val_losses.npy")[-1].mean()
            except FileNotFoundError:
                continue
        else:
            raise ValueError("metric must be either 'dice' or 'loss'")

        params = yaml.safe_load(open(dir_ / "config.yaml"))

        runs.append(
            RunInfo(
                score,
                params["learning_rate"],
                params["model_params"]["n_initial_filters"],
                params["batch_size"],
                params["loss_options"]["alpha"],
                params["epochs"],
                1 - params["lr_lambda"],
            )
        )

    # Print the best params
    n = 5
    top_dice_scores = set(sorted([r.score for r in runs], reverse=True)[:n])
    for r, d in zip(runs, data_dirs):
        if r.score in top_dice_scores:
            print(r, d.name)

    return _plot_scores(runs)


def _plot_coarse():
    """
    Read the losses and indices from the coarse search,
    then plot them colour coded by LR

    """
    paths = sorted(
        list((pathlib.Path(__file__).parents[1] / "tuning_output" / "coarse").glob("*"))
    )

    out_dir = pathlib.Path(__file__).parents[1] / "tuning_plots" / "coarse"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    fig = _lr_plot(paths)
    fig.savefig(out_dir / "coarse_search.png")
    plt.close(fig)

    fig = _filter_plot(paths)
    fig.savefig(out_dir / "coarse_search_n_filters.png")
    plt.close(fig)

    fig = _batch_plot(paths)
    fig.savefig(out_dir / "coarse_search_batch.png")
    plt.close(fig)

    fig = _plot_scatters(
        pathlib.Path(__file__).parents[1] / "tuning_output" / "coarse", metric="loss"
    )
    fig.suptitle(
        "NB: only run for a few epochs, minimum loss might mean LR is too high"
    )
    fig.savefig(str(out_dir / "scores.png"))


def _plot_med():
    """
    Read the losses and indices from the med,
    then plot them colour coded by LR and n filters

    """
    paths = sorted(
        list((pathlib.Path(__file__).parents[1] / "tuning_output" / "med").glob("*"))
    )

    out_dir = pathlib.Path(__file__).parents[1] / "tuning_plots" / "med"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    fig = _lr_plot(paths)
    fig.savefig(out_dir / "med_search.png")
    plt.close(fig)

    fig = _filter_plot(paths)
    fig.savefig(out_dir / "med_search_n_filters.png")
    plt.close(fig)

    fig = _batch_plot(paths)
    fig.savefig(out_dir / "med_search_batch.png")
    plt.close(fig)

    fig = _plot_scatters(
        pathlib.Path(__file__).parents[1] / "tuning_output" / "med", metric="dice"
    )
    fig.savefig(out_dir / "scores.png")


def _dicescore(results_dir: pathlib.Path) -> float:
    """
    Get the DICE score from the i-th run

    """
    dice_file = results_dir / "dice.txt"
    if not dice_file.exists():
        pred = np.load(results_dir / "val_pred.npy")
        truth = np.load(results_dir / "val_truth.npy").squeeze()

        # The prediction should already be scaled to be between 0 and 1
        if not pred.min() >= 0 and pred.max() <= 1:
            raise ValueError("Prediction should be scaled to between 0 and 1")

        # Get the DICE score
        score = metrics.dice_score(truth, pred)

        with open(dice_file, "w") as f:
            f.write(str(score))

    with open(dice_file) as f:
        return float(f.read().strip())


def _plot_scores(run_infos: list[RunInfo]) -> plt.Figure:
    """
    Plot histograms of the DICE scores and scatter plots

    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    axes[0, 0].hist([run.score for run in run_infos], bins=20)
    axes[0, 0].set_title("Scores")

    # Identify the top n
    n = 5
    scores = [run.score for run in run_infos]
    top_scores = set(sorted(scores, reverse=True)[:n])

    for axis, field in zip(axes.flat[1:], fields(RunInfo)[1:]):
        attr_name = field.name
        axis.plot([getattr(run, attr_name) for run in run_infos], scores, ".")
        axis.set_title(attr_name)

        # Plot the top N again in a different colour
        axis.plot(
            [getattr(run, attr_name) for run in run_infos if run.score in top_scores],
            [run.score for run in run_infos if run.score in top_scores],
            "r.",
        )

    # Log scale for learning rate and lambda
    axes[0, 1].set_xscale("log")
    axes[2, 0].set_xscale("log")

    # Turn the other axes off
    for axis in axes.flat[-2:]:
        axis.axis("off")

    fig.suptitle(f"N runs {len(run_infos)}")

    return fig


def _plot_fine():
    """
    Find the DICE accuracy of each, plot it

    """
    fig = _plot_scatters(
        pathlib.Path(__file__).parents[1] / "tuning_output" / "fine", metric="dice"
    )

    out_dir = pathlib.Path(__file__).parents[1] / "tuning_plots" / "fine"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    fig.savefig(str(out_dir / "scores.png"))


def main(mode: str):
    if mode == "coarse":
        _plot_coarse()
    elif mode == "med":
        _plot_med()
    else:
        _plot_fine()


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
