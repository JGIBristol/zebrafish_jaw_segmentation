"""
Perform inference on an out-of-sample subject

"""

import pathlib
import argparse

import torch
import tifffile
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.model import model, data
from fishjaw.images import metrics
from fishjaw.visualisation import images_3d
from fishjaw.inference import read, mesh


def _subject(config: dict, args: argparse.Namespace) -> tio.Subject:
    """
    Either read the image of choice and turn it into a Subject, or load the testing subject

    """
    return (
        read.test_subject(config["model_path"])
        if args.test
        else read.inference_subject(config, args.subject)
    )


def _mesh_projections(stl_mesh: mesh.Mesh) -> plt.Figure:
    """
    Visualize the mesh from three different angles

    """
    vertices = stl_mesh.vectors.reshape(-1, 3)
    faces = np.arange(vertices.shape[0]).reshape(-1, 3)

    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))

    plot_kw = {"cmap": "bone_r", "edgecolor": "k", "lw": 0.05}
    # First subplot: view from the front
    axes[0].plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        **plot_kw,
    )
    axes[0].view_init(elev=0, azim=0)

    # Second subplot: view from the top
    axes[1].plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        **plot_kw,
    )
    axes[1].view_init(elev=90, azim=0)

    # Third subplot: view from the side
    axes[2].plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        **plot_kw,
    )
    axes[2].view_init(elev=0, azim=90)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.tight_layout()
    return fig


def _save_mesh(
    segmentation: np.ndarray, subject_name: str, threshold: float, out_dir: pathlib.Path
) -> None:
    """
    Turn a segmentation into a mesh and save it

    """

    # Save as STL
    stl_mesh = mesh.cubic_mesh(segmentation > threshold)

    stl_mesh.save(f"inference/{subject_name}_mesh_{threshold:.3f}.stl")

    # Save projections
    fig = _mesh_projections(stl_mesh)
    fig.savefig(f"{out_dir}/{subject_name}_mesh_{threshold:.3f}_projections.png")
    plt.close(fig)


def _make_plots(
    args: argparse.Namespace,
    net: torch.nn.Module,
    subject: tio.Subject,
    config: dict,
    activation: str,
) -> None:
    """
    Make the inference plots using a model and subject

    """
    # Create the output dir
    out_dir, _ = args.model_name.split(".pkl")
    assert not _, "Model name should end with .pkl"

    out_dir = files.script_out_dir() / "inference" / out_dir
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # Perform inference
    prediction = model.predict(
        net,
        subject,
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=activation,
    )

    # Convert the image to a 3d numpy array - for plotting
    image = subject[tio.IMAGE][tio.DATA].squeeze().numpy()

    # Save the image and prediction as tiffs
    prefix = args.subject if args.subject else "test"
    tifffile.imwrite(out_dir / f"{prefix}_image.tif", image)
    tifffile.imwrite(out_dir / f"{prefix}_prediction.tif", prediction)

    # Save the output image and prediction as slices
    fig, _ = images_3d.plot_slices(image, prediction)

    # If we're using the test data, we have access to the ground truth so can
    # work out the Dice score and stick it in the plot too
    if args.test:
        truth = subject[tio.LABEL][tio.DATA].squeeze().numpy()
        dice = metrics.dice_score(truth, prediction)
        fig.suptitle(f"Dice: {dice:.3f}", y=0.99)

        # We might as well save the truth as a tiff too
        tifffile.imwrite(out_dir / f"{prefix}_truth.tif", truth)

        # Print a table of metrics
        print(metrics.table([truth], [prediction]).to_markdown())

    else:
        fig.suptitle(f"Inference: ID {args.subject}", y=0.99)

    fig.savefig(out_dir / f"{prefix}_slices.png")
    plt.close(fig)

    # Save the mesh
    if args.mesh:

        for threshold in np.arange(0.1, 1, 0.1):
            _save_mesh(prediction, prefix, threshold, out_dir)

        if args.test:
            # Mesh the ground truth too
            _save_mesh(truth, f"{prefix}_truth", 0.5, out_dir)


def _inference(args: argparse.Namespace, net: torch.nn.Module, config: dict) -> None:
    """
    Do the inference, save the plots

    """
    # Find which activation function to use from the config file
    # This assumes this was the same activation function used during training...
    activation = model.activation_name(config)

    # Either iterate over all subjects or just do the one
    if args.all:
        # Bad, should read these from a single place
        for subject in [273, 274, 218, 219, 120, 37]:
            print(f"Performing inference on subject {subject}")
            args.subject = subject
            _make_plots(args, net, _subject(config, args), config, activation)
    else:
        _make_plots(args, net, _subject(config, args), config, activation)


def main(args):
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    if args.subject == 247:
        raise RuntimeError("I think this one was in the training dataset...")

    # Load the model and training-time config
    model_state = model.load_model(args.model_name)

    config = model_state.config
    net = model_state.load_model(set_eval=True)

    net.to("cuda")

    _inference(args, net, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "subject",
        nargs="?",
        help="The subject to perform inference on",
        choices={247, 273, 274, 218, 219, 120, 37},
        type=int,
    )
    group.add_argument(
        "--test", help="Perform inference on the test data", action="store_true"
    )
    group.add_argument(
        "--all", help="Perform inference on all subjects", action="store_true"
    )

    parser.add_argument("--mesh", help="Save the mesh", action="store_true")
    parser.add_argument(
        "model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
    )

    main(parser.parse_args())
