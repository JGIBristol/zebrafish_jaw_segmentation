"""
Perform an ablation study on the requested data.

This will run the model on some data with and without the attention mechanism enabled-
we replace the attention block with the identity function, effectively disabling it.

Then makes some plots showing the results with and without the attention mechanism.

"""

import argparse

import torch
import torchio as tio
import matplotlib.pyplot as plt
from monai.networks.nets.attentionunet import AttentionBlock, AttentionUnet

from fishjaw.model import model, data
from fishjaw.inference import read


def ablated_psi(module, input, output):
    """
    Identity function to replace the attention mechanism

    """
    return torch.ones_like(output)


def _plot(
    net: torch.nn.Module,
    config: dict,
    inference_subject: tio.Subject,
    attention: bool,
    ax: tuple[plt.Axes, plt.Axes, plt.Axes],
) -> None:
    """
    Run inference and plot the results on the provided axes

    """
    # Remove the attention mechanism
    if not attention:
        for module in net.modules():
            if isinstance(module, AttentionBlock):
                psi_found = False
                for name, submodule in module.named_children():
                    if name == "psi":
                        if psi_found:
                            raise ValueError("Found multiple psi submodules")
                        submodule.register_forward_hook(ablated_psi)
                        psi_found = True

                if not psi_found:
                    raise ValueError("Couldn't find the psi submodule")

    # Perform inference
    prediction = model.predict(
        net,
        inference_subject,
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=model.activation_name(config),
    )

    # Plot it

    # Put the attention mechanism back
    if not attention:
        for hook in net._forward_hooks.values():
            hook.remove()


def main(args: argparse.Namespace):
    """
    Get the model and data and run it with and without the attention mechanism.

    """
    # Load the model and training-time config
    model_state = model.load_model(args.model_name)
    config = model_state.config

    net = model_state.load_model(set_eval=True)
    net.to("cuda")

    # Check that the model is the monai AttentionUnet - otherwise the ablation here won't work
    if not isinstance(net, AttentionUnet):
        raise ValueError(
            f"This script only works with the monai AttentionUnet, not {type(net)}"
        )

    # Load the subject to perform inference on
    inference_subject = (
        read.inference_subject(config, args.subject)
        if args.subject
        else read.test_subject(config["model_path"])
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for attention, ax in zip([True, False], axes):
        _plot(net, config, inference_subject, attention, ax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference with and without attention"
    )

    parser.add_argument(
        "subject",
        nargs="?",
        help="The subject to perform inference on. Defaults to using the test data.",
        choices={273, 274, 218, 219, 120, 37},
        type=int,
        default=None,
    )
    parser.add_argument(
        "model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
    )

    main(parser.parse_args())
