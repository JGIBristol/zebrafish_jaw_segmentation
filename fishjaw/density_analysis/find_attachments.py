"""
Find the muscle attachments by looking for areas of higher density.
"""

import numpy as np


def remove_ball(img: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """
    Remove a ball of the given radius from a 3D image.

    Sets the pixel values to 0 in the neighbourhood of the centre; otherwise
    keeps them as they are in `img`.

    :param img: 3d array to remove a ball from
    :param centre: the centre of the ball to remove, in pixel co-ords (z, x, y).
                   As might be returned by np.unravel_index(np.argmax(img), img.shape)
    :param radius: radius of ball to remove.

    :return: the image, with the ball set to 0

    """
    assert img.ndim == 3
    assert len(centre) == 3

    z, y, x = np.ogrid[: img.shape[0], : img.shape[1], : img.shape[2]]

    mask = (
        (z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2
    ) <= radius**2

    return np.where(mask, 0, img)
