"""
Tests for stuff in the images.py directory

"""

from ...images import transform


def test_around_centre():
    """
    Check that we can properly read the bool from the csv

    """
    assert transform.around_centre(30)
    assert transform.around_centre(107)

    # TODO add one for the false ones


def test_find_coords():
    """
    Check that we can properly read the co-ords from the csv

    """
    assert transform.centre(30) == (1435, 161, 390)
