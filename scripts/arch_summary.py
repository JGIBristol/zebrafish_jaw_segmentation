"""
Summarise the architecture of the model - only works for the attention unet

"""

import argparse

import torch
from prettytable import PrettyTable
from monai.networks.nets import attentionunet

from fishjaw.model import model, data


def count_parameters(net: torch.nn.Module) -> None:
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
                attentionunet.ConvBlock,
                attentionunet.UpConv,
            ),
        ):
            tracker = ReceptiveFieldTracker(layer)
            setattr(net, name, tracker)
        elif isinstance(layer, torch.nn.Sequential):
            replace_layers_with_tracker(layer)
        else:
            replace_layers_with_tracker(layer)


def main(*, model_name: str):
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    # Load the model
    model_state: model.ModelState = model.load_model(model_name)

    net = model_state.load_model(set_eval=True)
    if not isinstance(net, attentionunet.AttentionUnet):
        raise ValueError("This script only works for the attention unet sorry")
    net.to("cuda")

    # Print the number of trainable parameters
    count_parameters(net)

    # Track the size of the receptive field throughout the model
    replace_layers_with_tracker(net)

    dummy_input = torch.randn(1, 1, *data.get_patch_size(model_state.config)).to("cuda")
    net(dummy_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise the architecture of the model"
    )
    parser.add_argument(
        "model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
    )

    main(**vars(parser.parse_args()))
