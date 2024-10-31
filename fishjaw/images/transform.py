"""
Operations for transforming images

"""

import math
import pathlib
from functools import cache

import numpy as np
import pandas as pd


class UnexpectedCropError(Exception):
    """
    Raised when the crop size is larger than the image.
    """

    def __init__(self, message="Unexpected crop size mismatch"):
        self.message = message
        super().__init__(self.message)


class CropOutOfBoundsError(Exception):
    """
    Raised when the crop region goes out of bounds
    """

    def __init__(self, x: str, start: int, end: int, shape: tuple[int, int, int]):
        super().__init__(
            f"{x.upper()} index is out of bounds: {start, end} with image shape {shape}"
        )


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
    jaw_centre = jaw_centres().loc[n, ["z", "x", "y"]]

    # The slicing should give us a series, but if there are multiple matches
    # on the index then we will get a DataFrame
    if isinstance(jaw_centre, pd.DataFrame):
        raise ValueError(
            f"Expected one row in jaw_centres csv for fish {n}, got {len(jaw_centre)}"
        )

    return tuple(int(x) for x in jaw_centre.values)


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


def start_and_end(
    location: int, crop_size: int, *, start_from_loc: bool = False
) -> tuple[int, int]:
    """
    Find the start and end of the crop along one dimension

    :param location: the reference point for the crop
    :param crop_size: the size of the crop window
    :param start_from_loc: whether to start from the location and crop backwards (True),
                           or to use the location as the centre of the crop window (False)

    :returns: the start of the crop window
    :returns: the end of the crop window

    """
    # Ceiling so that if the crop_size is odd, our crop is offset backwards by 1
    # (instead of being offset forwards by 1) which I think is right but also it
    # doesn't matter all that much Z is either in the middle, or at the start
    start = (
        location - crop_size if start_from_loc else location - math.ceil(crop_size / 2)
    )

    return start, start + crop_size


def crop_out_of_bounds(start: int, end: int, length: int) -> bool:
    """
    Check if the start or end of the crop region are out of bounds

    :param start: start of the crop window
    :param end: end of the crop window
    :param length: size of the image in the chosen dimension

    :returns: bool for whether the crop extends beyond the image in either the positive
              or negative directions

    """
    return start < 0 or end > length


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
    :raises ValueError: if the cropped array doesn't match the crop size, which should
             never happen but its here to prevent regressions
    :raises ValueError: if the crop size is larger than the image
    :raises CropOutOfBoundsError: if the crop region would go out of bounds

    """
    if any(x > y for x, y in zip(crop_size, img.shape)):
        raise ValueError("Crop size is larger than the image")

    bounds = [
        start_and_end(a, b, start_from_loc=c)
        for (a, b, c) in zip(co_ords, crop_size, [not centred, False, False])
    ]

    for (start, end), length, x in zip(bounds, img.shape, "zxy"):
        if crop_out_of_bounds(start, end, length):
            raise CropOutOfBoundsError(x, start, end, img.shape)

    # TODO find out what is going on here
    # It should be ZXY but it looks like its ZYX (?)
    retval = img[
        bounds[0][0] : bounds[0][1],
        bounds[2][0] : bounds[2][1],
        bounds[1][0] : bounds[1][1],
    ]

    if retval.shape != crop_size:
        raise UnexpectedCropError(
            f"Expected cropped image to be {crop_size}, got {retval.shape}"
        )

    return retval
