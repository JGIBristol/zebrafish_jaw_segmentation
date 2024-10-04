"""
Test that we can create and train a model from the config file

"""

from functools import cache

from ...util import util
from ...model import model


@cache
def _config() -> dict:
    """
    Read the config file
    Defined here as a cached function to avoid reading the file multiple times

    """
    return util.userconf()


def test_create_model() -> None:
    """
    Check that we can create a model from the config file

    """
    # Read the config file
    config = _config()

    # Create a model from it
    model.model(config["model_params"])


def test_train_model() -> None:
    """
    Check that we can train a model, at least for a bit

    """
    # Read config file
    config = _config()

    # Create a model from it
    # Create some toy data
    # Train the model for 1 epoch
