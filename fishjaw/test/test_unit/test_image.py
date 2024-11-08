"""
Tests for stuff in the images.py directory

"""

import pytest
import numpy as np

from ...images import transform, metrics


@pytest.fixture
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


def test_fpr_binary(binary_images: tuple[np.ndarray, np.ndarray]):
    """
    Check that we can calculate the false positive rate

    """
    assert metrics.fpr(*binary_images) == 0.12


def test_tpr_binary(binary_images: tuple[np.ndarray, np.ndarray]):
    """
    Check that we can calculate the true positive rate

    """
    assert metrics.tpr(*binary_images) == 0.64
