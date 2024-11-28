"""
Integration tests for the model

"""

from ...model import model


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
