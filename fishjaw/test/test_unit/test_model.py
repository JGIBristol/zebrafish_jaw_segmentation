"""
Stuff for the model

"""

from ...model import model


def test_channels():
    """
    Check we get the right number of channels

    """
    assert model.channels(6, 3) == [3, 6, 12, 24, 48, 96]
