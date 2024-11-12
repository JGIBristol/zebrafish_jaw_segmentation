"""
Tests for stuff in the images.py directory

"""

import pytest
import numpy as np

from ...images import transform, metrics


@pytest.fixture(name="binary_images_")
def binary_images():
    """
    Two 10x10 binary images, both containing a 5x5 square of 1s
    that is offset by one in the x and y directions

    """
    image1 = np.zeros((10, 10), dtype=int)
    image1[2:7, 2:7] = 1

    image2 = np.zeros((10, 10), dtype=int)
    image2[3:8, 3:8] = 1

    return image1, image2


def test_get_window_size():
    """
    Check that we can correctly get the window size from config

    """
    config = {"window_size": "2,3,4"}
    assert transform.window_size(config) == (2, 3, 4)


def test_fpr_binary(binary_images_: tuple[np.ndarray, np.ndarray]):
    """
    Check that we can calculate the false positive rate

    """
    assert metrics.fpr(*binary_images_) == 0.12


def test_tpr_binary(binary_images_: tuple[np.ndarray, np.ndarray]):
    """
    Check that we can calculate the true positive rate

    """
    assert metrics.tpr(*binary_images_) == 0.64


def test_hausdorff():
    """
    Check we get the hausdorff distance right

    """
    shape = 10, 10

    x = np.zeros(shape)
    x[0, 0] = 1
    x[-1, 0] = 1

    y = np.zeros(shape)
    y[-3:, -3:] = 1

    assert np.isclose(metrics.hausdorff_distance(x, y), 7 * np.sqrt(2) / np.sqrt(200))
    assert np.isclose(metrics.hausdorff_distance(y, x), 7 * np.sqrt(2) / np.sqrt(200))


def test_hausdorff_3d():
    """
    Check we get the hausdorff distance right

    """
    shape = 3, 3, 3

    x = np.zeros(shape)
    x[0, 0, 0] = 1

    y = np.zeros(shape)
    y[2, 1, 1] = 1

    assert np.isclose(metrics.hausdorff_distance(x, y), np.sqrt(6) / np.sqrt(27))
    assert np.isclose(metrics.hausdorff_distance(y, x), np.sqrt(6) / np.sqrt(27))
