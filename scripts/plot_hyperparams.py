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

from fishjaw.util import files


@dataclass
class RunInfo:
    """
    The interesting parameters and results from a run

    """

    score: float
    lr: float
    n_filters: int
    batch_size: int
    alpha: float
    epochs: int
    one_minus_lambda: float


def _batch_plot(paths: list[pathlib.Path]) -> plt.Figure:
    """
    Plot the training and validation losses, colour coded by batch size

    """
    fig, axes = plt.subplots(1, 2, sharey=True)
    cmap = plt.get_cmap("viridis")

    for path in tqdm(paths):
        try:
            train_loss = np.load(path / "train_losses.npy").mean(axis=1)
            val_loss = np.load(path / "val_losses.npy").mean(axis=1)
        except FileNotFoundError:
            continue

        # Get the batch size
        with open(path / "config.yaml", encoding="utf-8") as f:
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
        with open(path / "config.yaml", encoding="utf-8") as f:
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
        with open(path / "config.yaml", encoding="utf-8") as f:
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

        with open(dir_ / "config.yaml", encoding="utf-8") as f:
            params = yaml.safe_load(f)

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


def _plot_coarse(input_dir: pathlib.Path, output_dir: pathlib.Path):
    """
    Read the losses and indices from the coarse search,
    then plot them colour coded by LR

    """
    paths = sorted(list((input_dir).glob("*")))

    fig = _lr_plot(paths)
    fig.savefig(output_dir / "coarse_search.png")
    plt.close(fig)

    fig = _filter_plot(paths)
    fig.savefig(output_dir / "coarse_search_n_filters.png")
    plt.close(fig)

    fig = _batch_plot(paths)
    fig.savefig(output_dir / "coarse_search_batch.png")
    plt.close(fig)

    fig = _plot_scatters(input_dir, metric="loss")
    fig.suptitle(
        "NB: only run for a few epochs, minimum loss might mean LR is too high"
    )
    fig.savefig(str(output_dir / "scores.png"))


def _plot_med(input_dir: pathlib.Path, output_dir: pathlib.Path):
    """
    Read the losses and indices from the med,
    then plot them colour coded by LR and n filters

    """
    paths = sorted(list(input_dir.glob("*")))

    fig = _lr_plot(paths)
    fig.savefig(output_dir / "med_search.png")
    plt.close(fig)

    fig = _filter_plot(paths)
    fig.savefig(output_dir / "med_search_n_filters.png")
    plt.close(fig)

    fig = _batch_plot(paths)
    fig.savefig(output_dir / "med_search_batch.png")
    plt.close(fig)

    fig = _plot_scatters(input_dir, metric="dice")
    fig.savefig(output_dir / "scores.png")


def _dicescore(results_dir: pathlib.Path) -> float:
    """
    Get the DICE score from the i-th run

    """
    # We'll write the score to a file
    dice_file = results_dir / "dice.txt"

    # We might have already calculated it
    if not dice_file.exists():
        # Find how many images there are in the validation set
        n_val_imgs = len(list(results_dir.glob("*val_pred_*.npy")))
        assert n_val_imgs == len(list(results_dir.glob("val_truth_*.npy")))

        if n_val_imgs == 0:
            raise FileNotFoundError(f"No validation images found in {results_dir}")

        # Get the DICE score
        # We want to combine the Dice score for multiple images
        # So keep track of the total intersection and volume here
        intersection = 0
        volume = 0

        for i in range(n_val_imgs):
            pred = np.load(results_dir / f"val_pred_{i}.npy")
            truth = np.load(results_dir / f"val_truth_{i}.npy").squeeze()

            assert pred.shape == truth.shape, f"{pred.shape=} != {truth.shape=}"

            # The prediction should already be scaled to be between 0 and 1
            if not pred.min() >= 0 and pred.max() <= 1:
                raise ValueError("Prediction should be scaled to between 0 and 1")

            intersection += np.sum(pred * truth)
            volume += pred.sum() + truth.sum()
        score = 2.0 * intersection / volume

        with open(dice_file, "w", encoding="utf-8") as f:
            f.write(str(score))

    # Read the score from the file
    with open(dice_file, encoding="utf-8") as f:
        return float(f.read().strip())


def _plot_scores(run_infos: list[RunInfo]) -> plt.Figure:
    """
    Plot histograms of the DICE scores and scatter plots

    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    # Identify the top n
    n = 5
    scores = [run.score for run in run_infos]
    top_scores = set(sorted(scores, reverse=True)[:n])

    # Identify the top few
    top_chunk = set(sorted(scores, reverse=True)[: 2 * len(scores) // 5])

    # Histogram of scores
    _, bins, _ = axes[0, 0].hist([run.score for run in run_infos], bins=20, label="All")
    axes[0, 0].set_title("Scores")

    # Plot the top quintile
    axes[0, 0].hist(
        [run.score for run in run_infos if run.score in top_chunk],
        bins=bins,
        color="y",
        label="Top 40%",
    )

    # Plot the top n again
    axes[0, 0].hist(
        [run.score for run in run_infos if run.score in top_scores],
        bins=bins,
        color="r",
        label="Top 5",
    )
    axes[0, 0].legend()

    for axis, field in zip(axes.flat[1:], fields(RunInfo)[1:]):
        attr_name = field.name
        axis.plot([getattr(run, attr_name) for run in run_infos], scores, ".")
        axis.set_title(attr_name)

        # Plot the top N again in a different colour
        axis.plot(
            [getattr(run, attr_name) for run in run_infos if run.score in top_chunk],
            [run.score for run in run_infos if run.score in top_chunk],
            "y.",
        )
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
    fig.tight_layout()

    return fig


def _plot_fine(input_dir: pathlib.Path, output_dir: pathlib.Path):
    """
    Find the DICE accuracy of each, plot it

    """
    fig = _plot_scatters(input_dir, metric="dice")

    fig.savefig(str(output_dir / "scores.png"))


def main(mode: str):
    """Choose the granularity of the search to plot"""
    input_dir = files.script_out_dir() / "tuning_output" / mode
    output_dir = files.script_out_dir() / "tuning_plots" / mode
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if mode == "coarse":
        _plot_coarse(input_dir, output_dir)
    elif mode == "med":
        _plot_med(input_dir, output_dir)
    else:
        _plot_fine(input_dir, output_dir)


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
