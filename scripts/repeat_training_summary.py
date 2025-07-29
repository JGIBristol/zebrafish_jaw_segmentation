"""
Once we've trained lots of models and run the inference with all of them,
read the tables of metrics from the logs and build a table.

Then print some summary stats
"""

import pathlib
import argparse

import tifffile
import pandas as pd
import matplotlib.pyplot as plt

from fishjaw.inference import read
from fishjaw.util import files, util
from fishjaw.images import transform, metrics


def _dump_repeat_segmentation_metrics():
    """
    Compare the segmentations from Felix (and possibly others)
    to the ground truth segmentations, and dump a pandas dataframe
    with the metrics for each model to disk.
    """
    config = util.userconf()
    seg_dir = (
        files.rdsf_dir(config)
        / "1Felix and Rich make models"
        / "Human validation STL and results"
    )
    # Read in the ground truth
    felix = tifffile.imread(
        seg_dir / "felix take2" / "ak_97-FBowers_complete.labels.tif"
    )

    # Read in the other segmentations
    felix2 = tifffile.imread(
        seg_dir
        / "New segmentations all felix"
        / "segmentation 2"
        / "ak_97_fbowers_2.labels.tif"
    )
    felix3 = tifffile.imread(
        seg_dir
        / "New segmentations all felix"
        / "Segmentation 3"
        / "ak_97_fbowers_3.labels.tif"
    )

    felix, felix2, felix3 = (
        transform.crop(
            x, read.crop_lookup()[97], transform.window_size(config), centred=True
        )
        for x in (felix, felix2, felix3)
    )

    # Compare the segmentations and compute the metrics
    table = metrics.table(
        [felix] * 3,
        [felix, felix2, felix3],
        thresholded_metrics=True,
    )
    table["label"] = ["felix", "felix2", "felix3"]
    table.set_index("label", inplace=True)

    # Dump it to disk
    with open(files.repeat_training_result_table_path(), "wb") as f:
        table.to_pickle(f)


def extract_table_from_file(filepath: pathlib.Path) -> pd.DataFrame:
    """
    Read a markdown table from file

    """
    df = pd.read_csv(
        filepath, sep="|", engine="python", skipinitialspace=True, skiprows=7
    )
    df.columns = df.columns.str.strip()

    df = df.map(lambda s: s.strip() if isinstance(s, str) else s)

    # Drop broken row and cols
    df = df.drop(index=0)
    for col in df.columns:
        if "unnamed" in col.lower():
            df = df.drop(columns=[col])

    df = df.set_index("label")

    # Turn everything to floats
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def hists(final_df: pd.DataFrame, ref_df: pd.DataFrame) -> None:
    """
    Plot histograms of the Dice scores + HD for all the models
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for axis, label in zip(axes, ["Dice", "1-Hausdorff_0.5"]):
        axis.hist(final_df[label], bins=25, label="Models", color="#648FFF")
        axis.axvline(
            ref_df.loc["felix", label],
            linestyle="--",
            label="P1",
            color="#DC267F",
        )
        axis.axvline(
            ref_df.loc["tahlia", label],
            linestyle="--",
            label="P2",
            color="#FE6100",
        )
        axis.axvline(
            ref_df.loc["harry", label],
            linestyle="--",
            label="P3",
            color="#FFB000",
        )

        axis.set_yticks(range(3))

    axes[0].legend()

    axes[0].set_xlabel("Dice Score")
    axes[1].set_xlabel("Hausdorff Distance (normalised)")

    fig.tight_layout()
    fig.savefig(files.script_out_dir() / "repeat_hists.png")
    plt.close(fig)

    # Plot hte combined score
    fig, axis = plt.subplots()
    axis.hist(
        final_df["Hausdorff_Dice_0.5"],
        bins=25,
        label="Models",
        color="#648FFF",
    )
    axis.axvline(
        ref_df.loc["felix", "Hausdorff_Dice_0.5"],
        linestyle="--",
        label="P1",
        color="#DC267F",
    )
    axis.axvline(
        ref_df.loc["tahlia", "Hausdorff_Dice_0.5"],
        linestyle="--",
        label="P2",
        color="#FE6100",
    )
    axis.axvline(
        ref_df.loc["harry", "Hausdorff_Dice_0.5"],
        linestyle="--",
        label="P3",
        color="#FFB000",
    )
    axis.legend()
    axis.set_xlabel("Hausdorff Dice Score (0.5)")
    axis.set_yticks(range(3))

    fig.tight_layout()
    fig.savefig(files.script_out_dir() / "repeat_combined.png")
    plt.close(fig)


def main():
    """
    Read all the markdown tables, convert them to dataframes, do some checks
    and print a summary table.

    """
    log_dir = pathlib.Path("logs/")

    dfs = []
    for file in log_dir.glob("*inference.log"):
        df = extract_table_from_file(file)
        dfs.append(df)

    # Ensure felix, harry, tahlia are the same in all files
    ref_df = dfs[0].loc[["felix", "harry", "tahlia"]]
    for i, df in enumerate(dfs[1:], start=1):
        assert df.loc[["felix", "harry", "tahlia"]].equals(ref_df)

    # TODO the implementation should be refactored to have a separate dataframe
    # for each person's segmentations, so we can then compare them and run
    # stats on them more sensibly

    # Get a df of metrics for felix's repeated segmentations, and stick it on the
    # end of our reference dataframe
    if not files.repeat_training_result_table_path().exists():
        _dump_repeat_segmentation_metrics()
    with open(files.repeat_training_result_table_path(), "rb") as f:
        repeat_df = pd.read_pickle(f)
    assert repeat_df.loc["felix"].equals(ref_df.loc["felix"])

    ref_df = pd.concat([ref_df, repeat_df.loc[["felix2", "felix3"]]])

    # Create the final combined DataFrame for the different models
    final_df = ref_df.copy()
    for i, df in enumerate(dfs):
        inference_row = df.loc["inference"].copy()
        inference_row.name = f"inference_{i+1}"
        final_df = pd.concat([final_df, pd.DataFrame([inference_row])])
    final_df.drop(["felix", "harry", "tahlia"], inplace=True)

    print(final_df.describe().to_markdown())

    # Print 2.5% and 97.5% confidence intervals for Dice and HD
    print(final_df.quantile([0.025, 0.975]).to_markdown())

    hists(final_df, ref_df)

    # print the rows containing the median and maximum Hausdorff_Dice_0.5
    # Find the median value
    median = final_df["Hausdorff_Dice_0.5"].median()
    closest_to_median_idx = (final_df["Hausdorff_Dice_0.5"] - median).abs().idxmin()

    # Find the row containing it
    print(f"Closest to median score: {final_df.loc[closest_to_median_idx].name}")
    print("Max score:", final_df.loc[final_df["Hausdorff_Dice_0.5"].idxmax()].name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and summarize training metrics from model logs in logs/"
    )

    main(**vars(parser.parse_args()))
