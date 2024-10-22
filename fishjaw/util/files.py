"""
Stuff for manipulating files

"""

import pathlib

from . import util


def rdsf_dir(config: dict) -> pathlib.Path:
    """
    Get the directory where the RDSF is mounted

    :param config: the configuration, e.g. from userconf.yml
    :returns: Path to the directory
    """
    return pathlib.Path(config["rdsf_dir"])


def mask_dirs(config: dict) -> list:
    """
    Get the directories where the masks are stored

    :param config: the configuration, e.g. from userconf.yml
    :returns: List of paths to the directories

    """
    return [rdsf_dir(config) / mask_dir for mask_dir in util.config()["label_dirs"]]


def dicom_dirs() -> list:
    """
    Get the directories where the DICOMs are stored

    """


def image_path(mask_path: pathlib.Path) -> pathlib.Path:
    """
    Get the path to the corresponding image for a mask

    :param mask_path: Path to the mask
    :returns: Path to the image

    """


def wahab_3d_tifs_dir(config: dict) -> pathlib.Path:
    """
    Get the directory where Wahab's 3D tifs are stored

    :param config: the configuration, e.g. from userconf.yml
    :returns: Path to the directory

    """
    return rdsf_dir(config) / util.config()["wahabs_3d_tifs"]


def model_path() -> pathlib.Path:
    """
    Get the path to the cached model, as created by scripts/train_model.py

    This is intended to be used with the model.ModelState class, so that we
    can keep the model state (weights, biases), the optimiser state and the
    configuration used to initialise the model/define the architecture and
    training parameters all in one place.

    :returns: Path to the model

    """
    return pathlib.Path(__file__).parents[2] / "model" / "model_state.pkl"
