"""
Stuff for manipulating files

"""

import pathlib

from . import util


def dicom_dir() -> pathlib.Path:
    """
    Get the directory where the DICOMs are stored

    :returns: Path to the directory

    """
    return pathlib.Path(util.userconf()["dicom_dir"])


def wahab_3d_tifs_dir() -> pathlib.Path:
    """
    Get the directory where Wahab's 3D tifs are stored

    :returns: Path to the directory

    """
    return util.rdsf_dir() / util.config()["wahabs_3d_tifs"]


def wahab_labels_dir() -> pathlib.Path:
    """
    Get the directory where Felix's second set of labelled images are stored
    Should be used with wahab_3d_tifs_dir() to get the corresponding images

    :returns: Path to the directory

    """
    return util.rdsf_dir() / util.config()["wahab_labels_dir"]


def felix_labels_2_dir() -> pathlib.Path:
    """
    Get the directory where Felix's second set of labelled images are stored
    Should be used with wahab_3d_tifs_dir() to get the corresponding images

    :returns: Path to the directory

    """
    return util.rdsf_dir() / util.config()["felix_labels_dir_2"]


def model_path() -> pathlib.Path:
    """
    Get the path to the cached, as created by scripts/train_model.py

    :returns: Path to the model

    """
    return pathlib.Path(__file__).parents[2] / "model" / "state_dict.pth"
