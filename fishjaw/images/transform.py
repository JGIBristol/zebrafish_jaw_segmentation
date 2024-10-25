"""
Operations for transforming images

"""

import math
import pathlib
from functools import cache

import numpy as np
import pandas as pd


@cache
def jaw_centres() -> pd.DataFrame:
    """
    Read the location of the jaw centres from file

    """
    csv_path = pathlib.Path(__file__).parents[2] / "data" / "jaw_centres.csv"
    return pd.read_csv(csv_path, skiprows=3).set_index("n")


def centre(n: int) -> tuple[float, float, float]:
    """
    Get the centre of the jaw for a given fish

    """
    return tuple(int(x) for x in jaw_centres().loc[n, ["z", "x", "y"]].values)


def around_centre(n: int) -> bool:
    """
    Whether cropping should use the co-ords as the centre or boundary

    :param n: fish number (using Wahab's new n convention; i.e. this matches
              the fish in DATABASE/uCT/Wahab_clean_dataset/TIFS)

    :returns: whether to crop around the centre or from the given Z index

    """
    return jaw_centres().loc[n, "crop_around_centre"]


def window_size(config: dict) -> tuple[int, int, int]:
    """
    Get the size of the window to crop from a dict of config (e.g. userconf.yml)

    :param config: must contain "window_size" as a comma-separated string of numbers
    :returns: Tuple of the window size

    """
    return tuple(int(x) for x in config["window_size"].split(","))


def crop_around_centre(
    img: np.ndarray,
    jaw_centre: tuple[int, int, int],
    crop_size: tuple[int, int, int],
) -> np.ndarray:
    """
    Crop an image around a given centre

    """
    d, w, h = crop_size
    z, y, x = jaw_centre

    # Ceiling so that if the crop_size is odd, we start offset backwards
    # which I think is right but also it doesn't matter all that much
    z_start = z - math.ceil(d / 2)
    x_start = x - math.ceil(h / 2)
    y_start = y - math.ceil(w / 2)

    return img[z_start : z_start + d, x_start : x_start + h, y_start : y_start + w]


def crop_from_z(
    img: np.ndarray,
    jaw_coords: tuple[int, int, int],
    crop_size: tuple[int, int, int],
) -> np.ndarray:
    """
    Crop an image around the centre of (x, y) and from z to z + d

    :param img: The input image
    :param jaw_coords: The coordinates to crop from (z, y, x)
    :param crop_size: The size of the crop (d, w, h)

    :returns: The cropped image as a numpy array
    """
    d, w, h = crop_size
    z, y, x = jaw_coords

    return img[z - d : z, y - w // 2 : y + w // 2, x - h // 2 : x + h // 2]


def crop(
    img: np.ndarray,
    co_ords: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    centred: bool,
) -> np.ndarray:
    """
    Crop an image, either around the centre or from the given Z index

    :param img: The input image
    :param jaw_centre: The centre coordinates (z, y, x)
    :param crop_size: The size of the crop (d, w, h)
    :param centred: whether to crop around the co-ords (true), or from
                    the given Z-co-ord onwards (false)

    :returns: The cropped image as a numpy array
    :raises: ValueError if the cropped array doesn't match the crop size

    """
    if centred:
        retval = crop_around_centre(img, co_ords, crop_size)
    else:
        retval = crop_from_z(img, co_ords, crop_size)

    if retval.shape != crop_size:
        raise ValueError(
            f"Expected cropped image to be {crop_size}, got {retval.shape}"
        )

    return retval
