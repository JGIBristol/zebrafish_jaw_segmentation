"""
Summarise the architecture of the model

"""

import argparse

import torch

from fishjaw.util import files, util
from fishjaw.model import model


class ReceptiveFieldTracker(torch.nn.Module):
    """Track the receptive field of the model"""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        """Forward pass"""
        out = self.layer(x)
        print(f"Layer {self.layer.__class__.__name__}: Output Shape: {out.shape}")
        return out


def replace_layers_with_tracker(net: torch.nn.Module):
    """Replace the layers with the receptive field tracker"""
    for name, layer in net.named_children():
        if isinstance(
            layer,
            (
                torch.nn.Conv3d,
                torch.nn.MaxPool3d,
                torch.nn.AvgPool3d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            tracker = ReceptiveFieldTracker(layer)
            setattr(net, name, tracker)
        elif isinstance(layer, torch.nn.Sequential):
            replace_layers_with_tracker(layer)
        else:
            replace_layers_with_tracker(layer)


def _load_model(config: dict) -> torch.nn.Module:
    """
    Load the model from disk

    """
    net = model.model(config["model_params"])

    # Load the state dict
    path = files.model_path()
    state_dict = torch.load(path)

    net.load_state_dict(state_dict["model"])
    net.eval()

    return net


def main():
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    config = util.userconf()

    # Load the model
    net = _load_model(config)
    net.to("cuda")

    replace_layers_with_tracker(net)

    dummy_input = torch.randn(1, 1, 160, 160, 160).to("cuda")
    net(dummy_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise the architecture of the model"
    )

    # Not passing any arguments
    parser.parse_args()
    main()
