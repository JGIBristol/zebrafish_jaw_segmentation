"""
Investigate the effect of removing layers and/or skip connections on the segmentation

"""

import argparse

import torch
import numpy as np
import torch.utils
import torchio as tio
import matplotlib.pyplot as plt
from tqdm import tqdm

from fishjaw.util import files
from fishjaw.model import model, data
from fishjaw.inference import read, mesh
from fishjaw.visualisation import plot_meshes
from monai.networks.nets.attentionunet import AttentionLayer


def zero_block(
    module: torch.nn.Module, input_: torch.tensor, output: torch.tensor
) -> torch.tensor:
    """
    Zero the output of a block

    :param module: The block being hooked, to zero out
    :param input_: The input to the block
    :param output: The output of the block

    :returns: The output, zeroed; same shape as the original output

    """
    # pylint: disable=unused-argument
    return output * 0.001 + input_[0].expand_as(output) * 0.999


def _predict(
    net: torch.nn.Module,
    config: dict,
    inference_subject: tio.Subject,
) -> np.ndarray:
    """
    Find the model's prediction on the inference subject

    """
    return model.predict(
        net,
        inference_subject,
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=model.activation_name(config),
        batch_size=1,
    )


def _remove_layer(
    net: torch.nn.Module, config: dict, layer_index: int
) -> tuple[torch.nn.Module, torch.utils.hooks.RemovableHandle]:
    """
    Turn off the layer at the given index and return the network and hook

    """
    index = 0
    for module in net.modules():
        if isinstance(module, AttentionLayer):
            if index == layer_index:
                hook = net.register_forward_hook(zero_block)
                break
            index += 1
    else:
        raise ValueError(f"Layer {layer_index} not found")
    return net, hook


def main(*, subject: int, model_name: str, threshold: float):
    """
    Load in a model and the data, then evaluate with and without each layer/skip connection

    """
    if not model_name.endswith(".pkl"):
        raise ValueError("Model name must end with '.pkl'")

    out_dir = files.script_out_dir() / "remove_layers" / model_name[:-4] / str(subject)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # Load the model and training-time config
    model_state = model.load_model(model_name)
    config = model_state.config

    net = model_state.load_model(set_eval=True)
    net.to("cuda")

    # Load in the subject
    inference_subject = (
        read.inference_subject(config, subject)
        if subject
        else read.test_subject(config["model_path"])
    )

    # Perform inference with the full model
    prediction = _predict(net, config, inference_subject) > threshold
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"})
    plot_meshes.projections(axes, mesh.cubic_mesh(prediction))
    fig.tight_layout()
    fig.savefig(out_dir / "full_model.png")
    plt.close(fig)

    # Sucessively remove layers and skip connections
    for layer_index in tqdm(range(4, -1, -1)):
        net, hook = _remove_layer(net, config, layer_index)
        prediction = _predict(net, config, inference_subject) > threshold
        print(dict(zip(*np.unique(prediction, return_counts=True))))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"})
        try:
            co_ords = np.argwhere(prediction)
            for axis, elev, azim in zip(axes, [0, 0, 90], [0, 90, 0]):
                axis.scatter(
                    co_ords[:, 0], co_ords[:, 1], co_ords[:, 2], s=2, alpha=0.2
                )
                axis.view_init(elev=elev, azim=azim)
        except ValueError as e:
            print(
                f"Layer {layer_index} broke, probably because {dict(zip(*np.unique(prediction, return_counts=True)))};\n{e}"
            )
        fig.tight_layout()
        fig.savefig(out_dir / f"removed_{layer_index}.png")
        plt.close(fig)
        hook.remove()


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
    main(**vars(parser.parse_args()))
