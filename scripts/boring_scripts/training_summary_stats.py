"""
Plot summary stats for the training, testing and validation data

"""

import pathlib

import pandas as pd
import matplotlib.pyplot as plt

from fishjaw.util import files, util
from fishjaw.images.transform import jaw_centres


def _n_from_paths(paths: list[pathlib.Path]) -> list[int]:
    """
    Get the number of fish from the paths

    :param paths: list of paths
    :returns: number of fish

    """
    return [int(path.stem) for path in paths]


def _print_info(
    label: str,
    n: list[int],
    fish_info: pd.DataFrame,
    dicom_dirs: list[str] | None = None,
) -> None:
    """
    Print the fish info

    :param n: list of fish numbers
    :param fish_info: DataFrame with fish info

    """
    if label == "train":
        assert dicom_dirs is not None

    df_slice = fish_info[["age", "genotype", "strain"]].loc[n]
    end_str = f"{'=' * 80}\n"

    print(f"Summary stats for {label} set")
    if label in {"val", "test"}:
        print(df_slice.to_markdown())
        print(end_str)

    else:
        # For train set, Find average and 95% CI for age
        print(f"N: {len(n)}")
        print(
            f"Average age: {df_slice['age'].mean():.2f} mo [95% CI:"
            f"{df_slice['age'].quantile([0.025, 0.975]).values}]"
        )
        print(df_slice.groupby("age").size().sort_index(ascending=True).to_markdown())
        print(
            df_slice.groupby(["genotype", "strain"])
            .size()
            .sort_values(ascending=False)
            .to_markdown()
        )
        print(end_str)

        # Get the training set for each fish
        mapping = {}
        for d in dicom_dirs:
            path = pathlib.Path(d)

            # Get N from the name
            training_set_n = path.name.split("Training set ", maxsplit=1)[1]
            training_set_n = training_set_n.split(" ", maxsplit=1)[0]

            mapping[training_set_n] = _n_from_paths(list(path.glob("*.dcm")))

        # Add a row to the df_slice for the training set
        def get_training_set(n: int) -> str:
            for k, v in mapping.items():
                if n in v:
                    return k
            return "Unknown"

        df_slice["training_set"] = df_slice.index.map(get_training_set)

        # Plot a histogram of ages in training set
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        df_slice["age"].hist(ax=axes, bins=[0.5 + x for x in range(0, 37)])
        axes.grid(False)
        axes.set_title("Age distribution in training set")
        axes.set_xlabel("Age (months)")
        fig.tight_layout()
        out_dir = files.boring_script_out_dir() / "train_data"
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        fig.savefig(out_dir / f"age_distribution_{label}.png")


def main():
    """
    Get the ID numbers of the fish used for training, testing and validation;
    print the average age and their mutations

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
    _print_info("train", train, fish_info, config["dicom_dirs"])

    # TODO Value counts for mutations


if __name__ == "__main__":
    main()
