"""
Integration tests for the model

"""

from ...model import model


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
        "spatial_dims": 3,
        "n_classes": 2,  # n bones + background
        "in_channels": 1,  # Our images are greyscale
        "n_layers": 6,  # 6?
        "n_initial_filters": 8,  # 6 would be sensible
        "kernel_size": 3,
        "stride": 2,  # 2
        "dropout": 0.0,
    }

    # Will raise an exception if something has gone wrong
    model.model(in_params)
