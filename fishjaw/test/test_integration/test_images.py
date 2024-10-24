"""
Tests for stuff in the images.py directory

"""

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
