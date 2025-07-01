"""
Once we've trained lots of models and run the inference with all of them,
read the tables of metrics from the logs and build a table.

Then print some summary stats
"""

import pathlib
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from fishjaw.util import files


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
        if not df.loc[["felix", "harry", "tahlia"]].equals(ref_df):
            raise AssertionError(
                f"Mismatch in felix/harry/tahlia values in file: {files[i]}"
            )

    # Create the final combined DataFrame
    final_df = ref_df.copy()
    for i, df in enumerate(dfs):
        inference_row = df.loc["inference"].copy()
        inference_row.name = f"inference_{i+1}"
        final_df = pd.concat([final_df, pd.DataFrame([inference_row])])
    final_df.drop(["felix", "harry", "tahlia"], inplace=True)

    print(final_df.describe().to_markdown())

    hists(final_df, ref_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and summarize training metrics from model logs in logs/"
    )

    main(**vars(parser.parse_args()))
