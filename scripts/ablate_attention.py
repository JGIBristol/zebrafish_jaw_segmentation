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
from fishjaw.util import files
from fishjaw.inference import read, mesh
from fishjaw.visualisation import plot_meshes


def ablated_psi(module, input_, output):
    """
    Identity function to replace the attention mechanism

    """
    # pylint: disable=unused-argument
    return torch.ones_like(output)


def register_hooks(
    net: torch.nn.Module, indices: list[int]
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Register hooks to replace the attention mechanism with the identity function

    """
    hooks = []
    attention_block_index = 0
    for module in net.modules():
        if isinstance(module, AttentionBlock):
            for name, submodule in module.named_children():
                if name == "psi" and attention_block_index in indices:
                    hooks.append(submodule.register_forward_hook(ablated_psi))

            attention_block_index += 1

    return hooks


def _plot(
    net: torch.nn.Module,
    config: dict,
    inference_subject: tio.Subject,
    ax: tuple[plt.Axes, plt.Axes, plt.Axes],
    indices: tuple[int] | None = None,
) -> None:
    """
    Possibly disable some attention mechanism(s), Run inference
    and plot the results on the provided axes

    """
    # Remove the attention mechanism
    if indices is not None:
        hooks = register_hooks(net, indices)

    # Perform inference
    prediction = model.predict(
        net,
        inference_subject,
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=model.activation_name(config),
    )

    # Create and plot a mesh
    plot_meshes.projections(ax, mesh.cubic_mesh(prediction > 0.5))

    # Put the attention mechanism back
    if indices is not None:
        for hook in hooks:
            hook.remove()


def main(args: argparse.Namespace):
    """
    Get the model and data and run it with and without the attention mechanism.

    """
    if not args.model_name.endswith(".pkl"):
        raise ValueError("Model name must end with '.pkl'")

    out_dir = files.script_out_dir() / "ablation" / args.model_name[:-4]
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

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

    # the number of attention layers
    for idx in range(5):
        fig, axes = plt.subplots(
            2, 3, figsize=(15, 10), subplot_kw={"projection": "3d"}
        )

        # Plot with attention
        _plot(net, config, inference_subject, axes[0])
        axes[0, 0].set_zlabel("With attention")

        # Plot without attention
        _plot(net, config, inference_subject, axes[0], indices=(idx,))
        axes[1, 0].set_zlabel("Without attention")

        fig.suptitle(
            f"Ablation study - {'test fish' if args.subject is None else f'subject {args.subject}'}"
        )

        fig.tight_layout()
        fig.savefig(
            out_dir
            / f"ablation_{'test' if args.subject is None else args.subject}_{idx}.png"
        )
        plt.close(fig)


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
