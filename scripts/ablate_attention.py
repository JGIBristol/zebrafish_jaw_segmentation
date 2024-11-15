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
from monai.networks.nets.attentionunet import AttentionBlock

from fishjaw.model import model, data
from fishjaw.inference import read


def identity_hook(module, input, output):
    """
    Identity function to replace the attention mechanism

    """
    return input[1]


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
                module.register_forward_hook(identity_hook)

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

    # Load the subject to perform inference on
    inference_subject = (
        read.inference_subject(config, args.subject)
        if args.subject
        else read.test_subject(config["model_path"])
    )

    net = model_state.load_model(set_eval=True)
    net.to("cuda")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for attention, ax in zip([True, False], axes):
        _plot(net, config, inference_subject, attention, ax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
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
