"""
Plot the results from the hyperparam search

"""

import pathlib
import argparse
from typing import Iterable
from multiprocessing import Pool
from dataclasses import dataclass, fields

import yaml
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.images import metrics


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
    one_minus_lambda: float


def _plot_scatters(data_dir: pathlib.Path, metric: str) -> plt.Figure:
    """
    Plot scatter plots and a histogram of dice scores if they exist
    metric must either be "dice" or "loss"

    """
    data_dirs = list(data_dir.glob("*"))
    runs = []

    for dir_ in data_dirs:
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


def _write_metrics_file(results_dir: pathlib.Path) -> None:
    """
    Write files containing the validation metrics for each run, as markdown

    :param results_dir: The directory containing the results from a run

    """
    # We'll write the score to a file
    metrics_file = results_dir / "metrics.txt"

    if metrics_file.exists():
        return

    n_val_imgs = len(list(results_dir.glob("*val_pred_*.npy")))
    n_truth_imgs = len(list(results_dir.glob("*val_truth_*.npy")))
    # The run hasn't finished yet
    if n_val_imgs == 0 and n_truth_imgs == 0:
        return

    # Something weird has happened - maybe the run was interrupted,
    # or we got unlucky and only one file has been written
    if n_val_imgs != n_truth_imgs:
        raise ValueError(
            f"Number of validation images {n_val_imgs} != number of truth images {n_truth_imgs}"
        )

    # Load the arrays
    pred = [np.load(results_dir / f"val_pred_{i}.npy") for i in range(n_val_imgs)]
    truth = [np.load(results_dir / f"val_truth_{i}.npy") for i in range(n_val_imgs)]

    # Check them
    for p, t in zip(pred, truth):
        assert p.shape == t.shape, f"{p.shape=} != {t.shape=}"
        if not p.min() >= 0 and p.max() <= 1:
            raise ValueError("Prediction should be scaled to between 0 and 1")

    table = metrics.table(truth, pred)
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(table.to_markdown())


def _write_all_metrics_files(data_dirs: Iterable[pathlib.Path], n_procs: int) -> None:
    """
    Create metrics files for all the runs

    """
    dirs = list(data_dirs)

    with Pool(n_procs) as pool, tqdm.tqdm(
        total=len(dirs), desc="Creating metric tables"
    ) as pbar:
        for _ in pool.imap(_write_metrics_file, dirs):
            pbar.update()
            pbar.refresh()


def _metrics_df(metrics_file: pathlib.Path) -> pd.DataFrame:
    """
    Read the metrics files and return a DataFrame

    """
    df = (
        pd.read_table(
            metrics_file,
            sep="|",
            index_col=1,
            header=0,
            skipinitialspace=True,
        )
        .dropna(how="all", axis=1)
        .iloc[1:]
        .astype(float)
    )
    df.columns = df.columns.str.strip()
    return df


def _dicescore(results_dir: pathlib.Path) -> float:
    """
    Get the DICE score from the i-th run

    :param results_dir: The directory containing the results from a run

    """
    # We'll write the score to a file
    metrics_file = results_dir / "metrics.txt"

    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics file found in {results_dir}")

    # get the average DICE score from the metrics file
    df = _metrics_df(metrics_file)

    return df["Dice"].mean()


def _plot_scores(run_infos: list[RunInfo]) -> plt.Figure:
    """
    Plot histograms of the DICE scores and scatter plots

    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    # Identify the top n
    n = 5
    scores = [run.score for run in run_infos]
    top_scores = set(sorted(scores, reverse=True)[:n])

    # Identify the top few
    top_chunk = set(sorted(scores, reverse=True)[: 2 * len(scores) // 5])

    # Histogram of scores
    bins = np.linspace(0, 1, 21, endpoint=True)
    axes[0, 0].hist([run.score for run in run_infos], bins=bins, label="All")
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

    # Log scale for learning rate, lambda and alpha
    axes[0, 1].set_xscale("log")
    axes[2, 0].set_xscale("log")
    axes[2, 1].set_xscale("log")

    fig.suptitle(f"N runs {len(run_infos)}")
    fig.tight_layout()

    return fig


def main(mode: str):
    """Choose the granularity of the search to plot"""
    input_dir = files.script_out_dir() / "tuning_output" / mode
    output_dir = files.script_out_dir() / "tuning_plots" / mode
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    fig = _plot_scatters(input_dir, metric="dice" if mode == "fine" else "loss")
    fig.savefig(output_dir / "scores.png")


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
    parser.add_argument(
        "--n_procs",
        type=int,
        help="How many processes to spawn for file stuff when making fine-grained plots",
        default=None,
    )

    args = parser.parse_args()

    # We need to write the files holding the table of metrics
    if args.mode == "fine":
        _write_all_metrics_files(
            (files.script_out_dir() / "tuning_output" / "fine").glob("*"),
            args.n_procs,
        )

    main(args.mode)
