"""
Operations for transforming images

"""

import pathlib
from functools import cache

import numpy as np
import pandas as pd

from ..util import util


@cache
def _jaw_centres() -> pd.DataFrame:
    """
    Read the location of the jaw centres from file

    """
    csv_path = pathlib.Path(__file__).parents[2] / "data" / "jaw_centres.csv"
    return pd.read_csv(csv_path).set_index("n")


def centre(n: int) -> tuple[float, float, float]:
    """
    Get the centre of the jaw for a given fish

    """
    return tuple(int(x) for x in _jaw_centres().loc[n, ["z", "x", "y"]].values)


def _window_size() -> tuple[int, int, int]:
    """
    Get the size of the window to crop

    """
    return tuple(int(x) for x in util.userconf()["window_size"].split(","))


def crop(
    img: np.ndarray,
    centre: tuple[int, int, int],
) -> np.ndarray:
    """
    Crop an image around a given centre

    """
    d, w, h = _window_size()
    z, y, x = centre

    return img[
        z - d // 2 : z + d // 2, x - h // 2 : x + h // 2, y - w // 2 : y + w // 2
    ]
