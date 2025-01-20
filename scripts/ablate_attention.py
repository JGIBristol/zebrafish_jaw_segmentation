"""
Perform an ablation study on the requested data, assuming the model is a monai
AttentionUnet (which it probably is?).

This will run the model on some data with and without the attention mechanism enabled-
we replace the attention block with the identity function, effectively disabling it.

Then makes some plots showing the results with and without the attention mechanism.

"""

import pathlib
import argparse
from typing import Any

import torch
import numpy as np
import torchio as tio
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from monai.networks.nets.attentionunet import AttentionBlock, AttentionUnet

from fishjaw.model import model, data
from fishjaw.util import files
from fishjaw.inference import read, mesh
from fishjaw.visualisation import plot_meshes
from fishjaw.images import metrics


def plot_projections(
    net: torch.nn.Module,
    config: dict[str, Any],
    inference_subject: tio.Subject,
    args: argparse.Namespace,
    out_dir: pathlib.Path,
):
    """
    Create figures for showing the projections with and without attention

    """
    n_attention_blocks = sum(
        1 for module in net.modules() if isinstance(module, AttentionBlock)
    )
    to_ablate = [(i,) for i in range(n_attention_blocks)] + [  # Single layers
        # Pairs
        (i, j)
        for i in range(n_attention_blocks)
        for j in range(n_attention_blocks)
        if i != j
    ]

    # Perform inference with attention
    with_attention = _predict(net, config, inference_subject)
    projection_fig_ax = [
        plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"projection": "3d"})
        for _ in to_ablate
    ]

    dice_matrix = np.zeros((n_attention_blocks, n_attention_blocks))

    for indices, (fig, axes) in tqdm(
        zip(to_ablate, projection_fig_ax), total=len(to_ablate)
    ):
        # Plot with attention
        axes[0, 0].set_zlabel("With attention")
        plot_meshes.projections(
            axes[0], mesh.cubic_mesh(with_attention > args.threshold)
        )

        # Plot without attention
        without_attention = _predict(net, config, inference_subject, indices=indices)
        axes[1, 0].set_zlabel("Without attention")
        plot_meshes.projections(
            axes[1], mesh.cubic_mesh(without_attention > args.threshold)
        )

        # Find the Dice similarity between them
        dice = metrics.float_dice(with_attention, without_attention)
        if len(indices) == 1:
            dice_matrix[indices[0], indices[0]] = dice
        if len(indices) == 2:
            dice_matrix[indices] = dice
            dice_matrix[indices[::-1]] = dice

        fig.suptitle(
            f"Ablation study - {'test fish' if args.subject is None else f'subject {args.subject}'}"
            f"\nDice similarity: {dice:.4f}"
            f"\nRemoved attention blocks: {indices}"
            f"\nThresholded at {args.threshold}",
        )

        fig.tight_layout()
        fig.savefig(
            out_dir / f"ablation_{'test' if args.subject is None else args.subject}"
            f"_{'_'.join(str(index) for index in indices)}.png"
        )
        plt.close(fig)

    return dice_matrix


def ablated_psi(module, input_, output) -> torch.Tensor:
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


def _predict(
    net: torch.nn.Module,
    config: dict,
    inference_subject: tio.Subject,
    indices: tuple[int] | None = None,
    batch_size: int = 1,
) -> np.ndarray:
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
        batch_size=batch_size,
    )

    # Put the attention mechanism back
    if indices is not None:
        for hook in hooks:
            hook.remove()

    return prediction


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

    # Choose which layers(and combinations of indices) to ablate
    dice_matrix = plot_projections(net, config, inference_subject, args, out_dir)

    # Heatmap of Dice similarities
    fig, axis = plt.subplots()

    im = axis.imshow(dice_matrix, cmap="plasma_r", vmin=0, vmax=1)
    axis.set_xlabel("Removed attention block(s)")
    axis.set_ylabel("Removed attention block(s)")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    fig.tight_layout()
    fig.savefig(
        out_dir / f"dice_heatmap_{'test' if args.subject is None else args.subject}.png"
    )


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarising the output (needed for meshing)",
    )

    main(parser.parse_args())
