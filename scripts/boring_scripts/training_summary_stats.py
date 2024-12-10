"""
Plot summary stats for the training, testing and validation data

"""
import pathlib

from fishjaw.util import files, util


def _n_from_paths(paths: list[pathlib.Path]) -> int:
    """
    Get the number of fish from the paths

    :param paths: list of paths
    :returns: number of fish
    """


def main():
    """
    Get the ID numbers of the fish used for training, testing and validation; print the average age and
    their mutations

    """
    # Read config
    config = util.userconf()

    val_files = files.dicom_paths(config, "val")
    test_files = files.dicom_paths(config, "test")
    train_files = files.dicom_paths(config, "train")

    # Get n from
    # files.dicom_paths(config: dict[str, Any], mode: str) -> list[pathlib.Path]:
    # For val set, just print the rows
    # For test set, print the row
    # For train set, Find average and 95% CI for age
    # Value counts for mutations


if __name__ == "__main__":
    main()
