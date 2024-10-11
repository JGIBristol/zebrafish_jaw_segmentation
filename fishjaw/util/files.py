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


def dicom_dir(config: dict) -> pathlib.Path:
    """
    Get the directory where the DICOMs are stored

    :param config: the configuration that tells us where the DICOMs are stored.
                   Might e.g. be from userconf.yml
    :returns: Path to the directory

    """
    return pathlib.Path(config["dicom_dir"])


def wahab_3d_tifs_dir(config: dict) -> pathlib.Path:
    """
    Get the directory where Wahab's 3D tifs are stored

    :param config: the configuration, e.g. from userconf.yml
    :returns: Path to the directory

    """
    return rdsf_dir(config) / util.config()["wahabs_3d_tifs"]


def wahab_labels_dir(config: dict) -> pathlib.Path:
    """
    Get the directory where Felix's second set of labelled images are stored
    Should be used with wahab_3d_tifs_dir() to get the corresponding images

    :param config: the configuration, e.g. from userconf.yml
    :returns: Path to the directory

    """
    return rdsf_dir(config) / util.config()["wahab_labels_dir"]


def felix_labels_2_dir(config: dict) -> pathlib.Path:
    """
    Get the directory where Felix's second set of labelled images are stored
    Should be used with wahab_3d_tifs_dir() to get the corresponding images

    :param config: the configuration, e.g. from userconf.yml
    :returns: Path to the directory

    """
    return rdsf_dir(config) / util.config()["felix_labels_dir_2"]


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
