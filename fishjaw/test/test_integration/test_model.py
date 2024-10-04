"""
Integration tests for the model

"""

from ...model import model


# How is this not a unit test
def test_channels() -> None:
    """
    Check we get the right sequence of channels from a given number
    of layers and number of filters per layer

    """
    assert model.channels(5, 25) == [32, 64, 128, 256, 512]


def test_model_params() -> None:
    """
    Check that the parameters for the model can be constructed correctly

    """
    in_params = {
        "model_name": "monai.networks.nets.AttentionUnet",
        "n_classes": 2,
        "n_layers": 6,
        "in_channels": 1,
        "spatial_dims": 3,
        "kernel_size": 3,
        "n_initial_filters": 14,
        "stride": 3,
        "dropout": 0.2,
    }

    # Will raise an exception if something has gone wrong
    model.model(in_params)
