"""
Find the muscle attachments by looking for areas of higher density.
"""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import pdist


def masked_smooth(
    img: np.ndarray, mask: np.ndarray, filter_size: int = 5
) -> np.ndarray:
    """
    Smooth an image, considering only the pixels in `img` selected by `mask`.

    Uses a mean filter.

    :param img: the image to smooth. Must only contain 0 outside of `mask`.
    :param mask: boolean array selecting which pixels to keep
    :param filter_size: size of the window used for smoothing

    :return: smoothed version of img
    :raises ValueError: if there are nonzero img pixels selected by mask
    """
    if not (img[~mask] == 0).all():
        raise ValueError("Img pixels outside the mask must be set to 0")

    sum = uniform_filter(img, size=filter_size)
    count = uniform_filter(mask.astype(np.float32), size=filter_size)

    # Avoid div by 0
    retval = np.divide(sum, count, where=count > 0)
    return retval * mask


def get_max_loc(arr: np.ndarray) -> np.ndarray:
    """
    Get the location of the maximum in an array.

    :param arr: an n-dimensional array.
    :return: an n-length array giving the location of the maximal element.
    """
    return np.unravel_index(np.argmax(arr, axis=None), arr.shape)


def _ball_mask(
    img_shape: tuple[int, int, int], centre: tuple[float, float, float], radius: float
) -> np.ndarray:
    """
    Binary mask for a ball at the given location/size in the provided sized array
    """
    z, y, x = np.ogrid[: img_shape[0], : img_shape[1], : img_shape[2]]

    return (
        (z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2
    ) <= radius**2


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

    return np.where(_ball_mask(img.shape, centre, radius), 0, img)


def get_maxima(
    image: np.ndarray, n_maxima: int, *, removal_radius: float
) -> list[np.ndarray]:
    """
    Get the locations of the top-n maxima from an image, in order.

    :param image: a 3-d image. Probably should be smoothed (see `masked_smooth`)
    :param n_maxima: number of maxima to find

    :return: a list of maxima locations, in descending order.

    """
    retval = []

    for _ in range(n_maxima):
        loc = get_max_loc(image)
        retval.append(loc)
        image = remove_ball(image, loc, removal_radius)

    return retval


def get_pairwise_distances(points: list[tuple]) -> np.ndarray:
    """
    Get all pairwise distances between a list of n-dimensional points
    """
    assert (
        len(set(len(x) for x in points)) == 1
    ), "all points must be same dimensionality"

    return pdist(points)
