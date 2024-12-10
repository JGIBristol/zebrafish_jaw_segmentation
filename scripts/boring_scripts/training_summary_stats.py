"""
Plot summary stats for the training, testing and validation data

"""

import pathlib

import pandas as pd
import prettytable

from fishjaw.util import files, util
from fishjaw.images.transform import jaw_centres


def _n_from_paths(paths: list[pathlib.Path]) -> list[int]:
    """
    Get the number of fish from the paths

    :param paths: list of paths
    :returns: number of fish

    """
    return [int(path.stem) for path in paths]


def _print_info(label: str, n: list[int], fish_info: pd.DataFrame) -> None:
    """
    Print the fish info

    :param n: list of fish numbers
    :param fish_info: DataFrame with fish info

    """
    df_slice = fish_info[["age"]].loc[n]

    print(f"Summary stats for {label} set")
    if label in {"val", "test"}:
        print(df_slice.to_markdown(), end=f"\n{'='*24}\n\n")

    else:
        # For train set, Find average and 95% CI for age
        print(
            f"\tAverage age: {df_slice['age'].mean():.2f} mo [95% CI: {df_slice['age'].quantile([0.025, 0.975]).values}]"
        )


def main():
    """
    Get the ID numbers of the fish used for training, testing and validation; print the average age and
    their mutations

    """
    # Read config
    config = util.userconf()
    fish_info = jaw_centres()

    val, test, train = (
        _n_from_paths(files.dicom_paths(config, mode))
        for mode in ["val", "test", "train"]
    )

    _print_info("val", val, fish_info)
    _print_info("test", test, fish_info)
    _print_info("train", train, fish_info)

    # Value counts for mutations


if __name__ == "__main__":
    main()
