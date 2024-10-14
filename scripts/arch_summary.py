"""
Summarise the architecture of the model

"""

import pickle
import argparse

import torch
from prettytable import PrettyTable

from fishjaw.util import files
from fishjaw.model import model


def count_parameters(net: torch.nn.Module):
    """Count the number of trainable parameters in the model"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")


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


def _load_model() -> torch.nn.Module:
    """
    Load the model from disk

    """

    # Load the state dict
    with open(str(files.model_path()), "rb") as f:
        model_state: model.ModelState = pickle.load(f)
    return model_state.load_model(set_eval=True)


def main():
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    # Load the model
    net = _load_model()
    net.to("cuda")

    # Print the number of trainable parameters
    print(count_parameters(net))

    # Track the size of the receptive field throughout the model
    replace_layers_with_tracker(net)
    # This should really use the architecture to find the size of the input
    # At the moment it's hard coded
    dummy_input = torch.randn(1, 1, 160, 160, 160).to("cuda")
    net(dummy_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise the architecture of the model"
    )

    # Not passing any arguments
    parser.parse_args()
    main()
