"""
Stuff for manipulating files

"""

import pathlib
from typing import Any

from . import util


def rdsf_dir(config: dict[str, Any]) -> pathlib.Path:
    """
    Get the directory where the RDSF is mounted

    :param config: the configuration, e.g. from userconf.yml
    :returns: Path to the directory
    """
    return pathlib.Path(config["rdsf_dir"])


def mask_dirs(config: dict[str, Any]) -> list:
    """
    Get the directories where the masks are stored

    :param config: the configuration, e.g. from userconf.yml
    :returns: List of paths to the directories

    """
    return [rdsf_dir(config) / mask_dir for mask_dir in util.config()["label_dirs"]]


def dicom_dirs(config: dict[str, Any]) -> list[pathlib.Path]:
    """
    Get the directories where the DICOMs are stored

    :param config: the configuration, e.g. from userconf.yml
    :returns: List of paths to the directories storing the DICOM files

    """
    return [pathlib.Path(dicom_dir) for dicom_dir in config["dicom_dirs"]]


def dicom_paths(config: dict[str, Any], mode: str) -> list[pathlib.Path]:
    """
    Get the paths to the DICOMs used for either training, validation or testing

    :param config: config, as might be read from userconf.yml
    :param mode: "train", "val", "test" or "all"

    :returns: a list of paths
    :raises ValueError: if mode does not match one of the expected values

    """
    if mode not in {"train", "val", "test", "all"}:
        raise ValueError(
            f"mode must be one of 'train', 'test', 'val' or 'all', not {mode}"
        )

    all_dicoms = [
        dicom for directory in dicom_dirs(config) for dicom in directory.glob("*.dcm")
    ]

    # Sanity check - there should be no duplicated DICOMs
    dicom_stems = [dicom.stem for dicom in all_dicoms]
    if len(dicom_stems) != len(set(dicom_stems)):
        raise RuntimeError(
            f"Duplicate DICOMs found: {set(x for x in dicom_stems if dicom_stems.count(x) > 1)}"
        )

    # Returning all is easy
    if mode == "all":
        return all_dicoms

    # Otherwise, we need to filter
    val_paths = config["validation_dicoms"]
    test_paths = config["test_dicoms"]

    # Check there's no overlap; you never know...
    if len(set(val_paths) & set(test_paths)):
        raise RuntimeError(
            f"Overlap between validation and test sets: {val_paths} & {test_paths}"
        )

    retval = []
    for path in all_dicoms:
        stem = path.stem

        # Training data is everything that isn't in the validation or test sets
        if mode == "train":
            if stem not in (val_paths + test_paths):
                retval.append(path)

        elif mode == "val":
            if stem in val_paths:
                retval.append(path)

        elif mode == "test":
            if stem in test_paths:
                retval.append(path)

    return retval


def image_path(mask_path: pathlib.Path) -> pathlib.Path:
    """
    Get the path to the corresponding image for a mask.
    These both live on the RDSF, so we just need to replace some directories with
    where Wahab stored his 3D tiffs.

    :param mask_path: Path to the mask
    :returns: Path to the image

    """
    # Take the name from mask_path
    file_name = mask_path.name.replace(".labels.tif", ".tif")

    # Remove the "ak_" from the start
    if not file_name.startswith("ak_"):
        raise ValueError(f"File name {file_name} does not start with 'ak_'")
    file_name = file_name[3:]

    # We've hard-coded the number of dirs to strip off which is bad - if we later move
    # the label_dirs to somewhere deeper/shallower on the RDSF, then it'll break,
    # but hopefully that won't happen
    return mask_path.parents[3] / util.config()["wahabs_3d_tifs"] / file_name


def wahab_3d_tifs_dir(config: dict[str, Any]) -> pathlib.Path:
    """
    Get the directory where Wahab's 3D tifs are stored

    :param config: the configuration, e.g. from userconf.yml
    :returns: Path to the directory

    """
    return rdsf_dir(config) / util.config()["wahabs_3d_tifs"]


def model_path(config: dict[str, Any]) -> pathlib.Path:
    """
    Get the path to the cached model, as created by scripts/train_model.py

    This is intended to be used with the model.ModelState class, so that we
    can keep the model state (weights, biases), the optimiser state and the
    configuration used to initialise the model/define the architecture and
    training parameters all in one place.

    :param config: the configuration, e.g. from userconf.yml.
                   Just needs to have a "model_path" key
    :returns: Path to the model. If the path doesn't end in .pkl, it will be appended

    """
    path = config["model_path"]
    if not path.endswith(".pkl"):
        path = path + ".pkl"

    return pathlib.Path(__file__).parents[2] / "model" / path


def script_out_dir() -> pathlib.Path:
    """
    Get the directory where the output of scripts is stored, creating it if it doesn't exist

    :returns: Path to the directory

    """
    retval = pathlib.Path(__file__).parents[2] / "script_output"
    if not retval.is_dir():
        retval.mkdir()
    return retval


def boring_script_out_dir() -> pathlib.Path:
    """
    Get the directory where the output of the extra scripts is stored,
    creating it if it doesn't exist

    :returns: Path to the directory

    """
    retval = pathlib.Path(__file__).parents[2] / "boring_script_output"
    if not retval.is_dir():
        retval.mkdir()
    return retval


def broken_dicoms() -> set[int]:
    """
    Get the IDs of the broken DICOMs

    """
    return set(util.config()["broken_ids"])


def duplicate_dicoms() -> set[int]:
    """
    Get the IDs of the broken DICOMs

    """
    return set(util.config()["duplicate_ids"])
