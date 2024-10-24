"""
Tests for stuff in the images.py directory

"""

from ...images import transform


def test_get_window_size():
    """
    Check that we can correctly get the window size from config

    """
    config = {"window_size": "2,3,4"}
    assert transform.window_size(config) == (2, 3, 4)
