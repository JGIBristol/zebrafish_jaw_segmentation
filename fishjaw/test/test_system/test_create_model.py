"""
Test that we can create and train a model from the config file

"""

from functools import cache

import torch
import torchio as tio

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
    # Create a model
    config = _config()
    net = model.model(config["model_params"])

    # Create some toy data
    window_size = [int(x) for x in config["window_size"].split(",")]
    subject = tio.Subject(
        image=tio.Image(tensor=torch.ones(1, *window_size), type=tio.INTENSITY),
        label=tio.Image(tensor=torch.ones(1, *window_size), type=tio.LABEL),
    )
    subjects_dataset = tio.SubjectsDataset([subject])

    # Create a dataconfig object
    data = model.DataConfig(config, subjects_dataset, subjects_dataset)

    # Train the model for 1 epoch
    loss_fn = model.lossfn(config)
    optim = model.optimiser(config, net)
    train_options = model.TrainingConfig(device="cpu", epochs=1)

    config["epochs"] = 1
    model.train(net, optim, loss_fn, data, train_options)
