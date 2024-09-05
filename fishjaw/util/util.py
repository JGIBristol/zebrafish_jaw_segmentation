"""
Very general utilities

"""

import yaml
import pathlib


def rootdir() -> pathlib.Path:
    """
    Root dir of this git repo

    """
    return pathlib.Path(__file__).parents[2]


def userconf() -> dict:
    """
    Get the user configuration

    :returns: The user configuration

    """
    with open(rootdir() / "userconf.yml", "r") as f:
        return yaml.safe_load(f)


def config() -> dict:
    """
    Get the global config

    :returns: The config

    """
    with open(rootdir() / "config.yml", "r") as f:
        return yaml.safe_load(f)


def rdsf_dir() -> pathlib.Path:
    """
    Get the directory where the RDSF is mounted

    :returns: Path to the directory
    """
    return pathlib.Path(userconf()["rdsf_dir"])
