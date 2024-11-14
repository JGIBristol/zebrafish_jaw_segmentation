"""
Plot the results from the hyperparam search

"""

import pathlib
import argparse
from typing import Iterable
from dataclasses import dataclass, fields

import yaml
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from fishjaw.util import files
from fishjaw.images import metrics
from fishjaw.visualisation import images_3d


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


def _write_metrics_file(results_dir: pathlib.Path) -> None:
    """
    Write files containing the validation metrics for each run, as markdown
    Also plots ROC curves for each validation image

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

    # Plot ROC curves and thresholded at 0.5 with largest component
    for i, (p, t) in enumerate(zip(pred, truth)):
        roc_path = results_dir / f"roc_curve_{i}.png"
        if not roc_path.exists():
            fig, axis = plt.subplots()
            try:
                fpr, tpr, threshold = roc_curve(t.ravel(), p.ravel())
            except ValueError as e:
                print(e)
                continue

            axis.plot(fpr, tpr)
            scatter = axis.scatter(fpr, tpr, c=threshold)

            cbar = fig.colorbar(scatter)
            cbar.set_label("Threshold")

            axis.set_xlabel("FPR")
            axis.set_ylabel("TPR")
            axis.set_title(
                f"ROC curve for validation image {i}: AUC = {auc(fpr, tpr):.3f}"
            )

            fig.tight_layout()
            fig.savefig(roc_path)
            plt.close(fig)

        # Make thresholded plots
        threshold_path = results_dir / f"thresholded_{i}.png"
        if not threshold_path.exists():
            thresholded = metrics.largest_connected_component(p > 0.5)
            fig, _ = images_3d.plot_slices(thresholded)
            fig.savefig(threshold_path)
            plt.close(fig)

    table = metrics.table(truth, pred)
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(table.to_markdown())


def _write_all_metrics_files(data_dirs: Iterable[pathlib.Path]) -> None:
    """
    Create metrics files for all the runs

    """
    dirs = list(data_dirs)

    for d in tqdm.tqdm(dirs):
        _write_metrics_file(d)


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


def _metric(results_dir: pathlib.Path, metric: str) -> float:
    """
    Get a metric score from the i-th run

    :param results_dir: The directory containing the results from a run

    """
    # We'll write the score to a file
    metrics_file = results_dir / "metrics.txt"

    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics file found in {results_dir}")

    # get the average DICE score from the metrics file
    df = _metrics_df(metrics_file)

    return df[metric].mean()


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


def _plot_scatters(data_dir: pathlib.Path, metric: str) -> plt.Figure:
    """
    Plot scatter plots and a histogram of dice scores if they exist
    metric must either be "dice" or "loss"

    """
    data_dirs = list(data_dir.glob("*"))
    runs = []

    for dir_ in data_dirs:
        if metric == "loss":
            try:
                score = 1 - np.load(dir_ / "val_losses.npy")[-1].mean()
            except FileNotFoundError:
                continue
        else:
            try:
                score = _metric(dir_, metric)
            except FileNotFoundError:
                continue

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
    print(f"Top {metric} scores:")
    n = 5
    top_dice_scores = set(sorted([r.score for r in runs], reverse=True)[:n])
    for r, d in zip(runs, data_dirs):
        if r.score in top_dice_scores:
            print(r, d.name)
    print("=" * 80)

    return _plot_scores(runs)


def main(mode: str, out_dir: str):
    """Choose the granularity of the search to plot"""
    input_dir = files.script_out_dir() / out_dir / mode
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory {input_dir} not found")

    output_dir = files.script_out_dir() / "tuning_plots" / out_dir / mode
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if mode != "fine":
        fig = _plot_scatters(input_dir, metric="loss")
        fig.savefig(output_dir / "scores.png")
    else:
        for metric in [
            "Dice",
            "1-FPR",
            "TPR",
            "Precision",
            "Recall",
            "Jaccard",
            "ROC AUC",
            "G_Measure",
            # "1-Hausdorff_0.5",
            # "Hausdorff_Dice_0.5",
            "Z_dist_score",
        ]:
            fig = _plot_scatters(input_dir, metric=metric)
            fig.savefig(output_dir / f"{metric}.png")
            plt.close(fig)


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
        "out_dir",
        type=str,
        help="Directory to read the tuning outputs from,"
        "relative to the script output directory",
    )

    args = parser.parse_args()

    # We need to write the files holding the table of metrics
    if args.mode == "fine":
        _write_all_metrics_files(
            sorted(
                list((files.script_out_dir() / args.out_dir / args.mode).glob("*")),
                key=lambda x: int(x.name),
            )
        )

    main(args.mode, args.out_dir)
