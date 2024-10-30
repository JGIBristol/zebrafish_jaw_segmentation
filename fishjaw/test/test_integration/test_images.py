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

    assert not transform.around_centre(25)


def test_find_coords():
    """
    Check that we can properly read the co-ords from the csv

    """
    assert transform.centre(30) == (1435, 161, 390)


@pytest.fixture(name="test_img")
def uniform_slices():
    """
    Fixture that provides a 10x10 3D array where each slice has uniform values.
    The first slice is all ones , the second slice is all twos, etc.
    """
    return np.arange(1, 11)[:, np.newaxis, np.newaxis] * np.ones(
        (10, 10, 10), dtype=int
    )


def test_central_crop_even_window(test_img: np.ndarray):
    """
    Check that we can correctly crop an image around the centre

    """
    centre = (5, 5, 5)
    crop_size = (4, 4, 4)

    # Expected output is an array with values 4, 5, 6, 7
    expected = np.arange(4, 8)[:, np.newaxis, np.newaxis] * np.ones(
        crop_size, dtype=int
    )

    cropped = transform.crop(test_img, centre, crop_size, centred=True)

    assert (cropped == expected).all()


def test_central_crop_odd_window(test_img: np.ndarray):
    """
    Check that we can correctly crop an image around the centre

    """
    centre = (5, 5, 5)
    crop_size = (3, 3, 3)

    # Expected output is an array with values 4, 5, 6
    expected = np.arange(4, 7)[:, np.newaxis, np.newaxis] * np.ones(
        crop_size, dtype=int
    )

    cropped = transform.crop(test_img, centre, crop_size, centred=True)

    assert (cropped == expected).all()


def test_offset_crop_even_window(test_img: np.ndarray):
    """
    Check that we can correctly crop an image up to a given slice

    """
    centre = (5, 5, 5)
    crop_size = (4, 4, 4)

    expected = np.arange(2, 6)[:, np.newaxis, np.newaxis] * np.ones(
        crop_size, dtype=int
    )

    cropped = transform.crop(test_img, centre, crop_size, centred=False)

    assert (cropped == expected).all()


def test_offset_crop_odd_window(test_img: np.ndarray):
    """
    Check that we can correctly crop an image up to a given slice

    """
    centre = (5, 5, 5)
    crop_size = (3, 3, 3)

    expected = np.arange(3, 6)[:, np.newaxis, np.newaxis] * np.ones(
        crop_size, dtype=int
    )

    cropped = transform.crop(test_img, centre, crop_size, centred=False)

    assert (cropped == expected).all()


def test_crop_larger_than_image():
    """
    Check that the right error is raised

    """
    with pytest.raises(ValueError):
        transform.crop(np.ones((10, 10, 10)), (5, 5, 5), (10, 10, 11), True)

    with pytest.raises(ValueError):
        transform.crop(np.ones((10, 10, 10)), (5, 5, 5), (10, 11, 10), True)

    with pytest.raises(ValueError):
        transform.crop(np.ones((10, 10, 10)), (5, 5, 5), (11, 10, 10), True)


def test_z_overlap(test_img: np.ndarray):
    """
    Check the right error is raised

    """
    crop_size = (4, 4, 4)
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (1, 5, 5), crop_size, True)

    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (9, 5, 5), crop_size, True)

    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (1, 5, 5), crop_size, False)

    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (11, 5, 5), crop_size, False)


def test_x_out_of_bounds_crop(test_img: np.ndarray):
    """
    Check the right error is raised

    """
    crop_size = (4, 4, 4)

    # Crop right up to the edge
    transform.crop(test_img, (5, 8, 5), crop_size, True)
    # Too large
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (5, 9, 5), crop_size, True)

    # Crop right from the start
    transform.crop(test_img, (5, 2, 5), crop_size, True)
    # Too small
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (5, 1, 5), crop_size, True)


def test_x_out_of_bounds_crop_offset(test_img: np.ndarray):
    """
    Check the right error is raised

    """
    crop_size = (4, 4, 4)
    transform.crop(test_img, (5, 8, 5), crop_size, False)
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (5, 9, 5), crop_size, False)
    transform.crop(test_img, (5, 2, 5), crop_size, False)
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (5, 1, 5), crop_size, False)


def test_y_out_of_bounds_crop(test_img: np.ndarray):
    """
    Check the right error is raised

    """
    crop_size = (4, 4, 4)

    # Crop right up to the edge
    transform.crop(test_img, (5, 5, 8), crop_size, True)
    # Too large
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (5, 5, 9), crop_size, True)

    # Crop right from the start
    transform.crop(test_img, (5, 5, 2), crop_size, True)
    # Too small
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (5, 5, 1), crop_size, True)


def test_y_out_of_bounds_crop_offset(test_img: np.ndarray):
    """
    Check the right error is raised

    """
    crop_size = (4, 4, 4)
    transform.crop(test_img, (5, 5, 8), crop_size, False)
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (5, 5, 9), crop_size, False)
    transform.crop(test_img, (5, 5, 2), crop_size, False)
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (5, 5, 1), crop_size, False)


def test_z_out_of_bounds_crop(test_img: np.ndarray):
    """
    Check the right error is raised

    """
    crop_size = (4, 4, 4)

    # Crop right up to the edge
    transform.crop(test_img, (8, 5, 5), crop_size, True)
    # Too large
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (9, 5, 5), crop_size, True)

    # Crop right from the start
    transform.crop(test_img, (2, 5, 5), crop_size, True)
    # Too small
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (1, 5, 5), crop_size, True)


def test_z_out_of_bounds_crop_offset(test_img: np.ndarray):
    """
    Check the right error is raised

    """
    crop_size = (4, 4, 4)
    transform.crop(test_img, (10, 5, 5), crop_size, False)
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (11, 5, 5), crop_size, False)
    transform.crop(test_img, (4, 5, 5), crop_size, False)
    with pytest.raises(transform.CropOutOfBoundsError):
        transform.crop(test_img, (3, 5, 5), crop_size, False)
