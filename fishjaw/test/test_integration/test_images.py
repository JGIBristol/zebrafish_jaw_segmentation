"""
Tests for stuff in the images.py directory

"""

import pytest
import numpy as np

from ...images import transform


def test_read_jaw_centres():
    """
    Check we can read the jaw location table

    """
    table = transform.jaw_centres()

    # Just check one row
    assert table.loc[30, "z"] == 1435
    assert table.loc[30, "x"] == 161
    assert table.loc[30, "y"] == 390
    assert table.loc[30, "crop_around_centre"]


def test_around_centre():
    """
    Check that we can properly read the bool from the csv

    """
    assert transform.around_centre(30)
    assert transform.around_centre(107)

    assert not transform.around_centre(1)


def test_find_coords():
    """
    Check that we can properly read the co-ords from the csv

    """
    assert transform.centre(30) == (1435, 161, 390)


@pytest.fixture
def uniform_slices():
    """
    Fixture that provides a 10x10 3D array where each slice has uniform values.
    The first slice is all zeros, the second slice is all ones, etc.
    """
    # Create a 3D array with shape (10, 10, 10)
    array = np.arange(10)[:, np.newaxis, np.newaxis] * np.ones((10, 10, 10), dtype=int)

    return array


def test_central_crop():
    """
    Check that we can correctly crop an image around the centre

    """
