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
from fishjaw.images import metrics, transform
from fishjaw.visualisation import images_3d, plot_meshes
from fishjaw.inference import read, mesh


def _subject(config: dict, args: argparse.Namespace) -> tio.Subject:
    """
    Either read the image of choice and turn it into a Subject, or load the testing subject

    """
    print("Reading subject")
    return (
        read.test_subject(config["model_path"])
        if args.test
        else read.inference_subject(config, args.subject)
    )


def _save_mesh(
    segmentation: np.ndarray,
    subject_name: str,
    threshold: float,
    out_dir: pathlib.Path,
) -> None:
    """
    Turn a segmentation into a mesh and save it

    """

    # Save as STL
    stl_mesh = mesh.cubic_mesh(segmentation > threshold)

    stl_mesh.save(f"inference/{subject_name}_mesh_{threshold:.3f}.stl")

    # Save projections
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))
    plot_meshes.projections(axes, stl_mesh, plot_kw={"alpha": 0.4, "cmap": "cividis_r"})

    fig.tight_layout()
    fig.savefig(
        f"{out_dir}/{subject_name}_mesh_{threshold:.3f}_projections.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def _save_test_meshes(
    thresholded_pred: np.ndarray,
    truth: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """
    Turn a segmentation into a mesh and save it

    """
    # Save the prediction and truth as STLs
    prediction_mesh = mesh.cubic_mesh(thresholded_pred)
    prediction_mesh.save(f"{out_dir}/test_mesh.stl")
    truth_mesh = mesh.cubic_mesh(truth)
    truth_mesh.save(f"{out_dir}/test_mesh_truth.stl")

    # Create the figure
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))

    # Find the Hausdorff points
    hausdorff_points = metrics.hausdorff_points(truth, thresholded_pred)

    # Make projections of the meshes
    plot_meshes.projections(
        axes,
        prediction_mesh,
        plot_kw={"alpha": 0.2, "color": "blue", "label": "Prediction"},
    )
    plot_meshes.projections(
        axes, truth_mesh, plot_kw={"alpha": 0.1, "color": "grey", "label": "Truth"}
    )

    # Indicate Hausdorff distance
    x, y, z = zip(*hausdorff_points)
    for ax in axes:
        ax.plot(x, y, z, "rx-", markersize=4, label="Hausdorff distance")

    axes[0].legend(loc="upper right")

    fig.savefig(
        f"{out_dir}/test_mesh_overlaid_projections.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def _make_plots(
    args: argparse.Namespace,
    net: torch.nn.Module,
    subject: tio.Subject,
    config: dict,
    activation: str,
    batch_size: int = 1,
) -> None:
    """
    Make the inference plots using a model and subject

    """
    # Create the output dir
    out_dir, _ = args.model_name.split(".pkl")
    assert not _, "Model name should end with .pkl"

    # Append _speckle if we're removing voxels
    if args.speckle:
        out_dir += "_speckle"
    if args.splotch:
        out_dir += "_splotch"

    out_dir = files.script_out_dir() / "inference" / out_dir
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # Perform inference
    print("Performing inference")
    prediction = model.predict(
        net,
        subject,
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=activation,
        batch_size=batch_size,
    )

    # Remove every second voxel to make the segmentation worse
    if args.speckle:
        prediction = transform.speckle(prediction)

    if args.splotch:
        prediction = transform.add_random_blobs(args.rng, prediction)

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
        print(
            metrics.table([truth], [prediction], thresholded_metrics=True).to_markdown()
        )

    else:
        fig.suptitle(f"Inference: ID {args.subject}", y=0.99)

    fig.savefig(out_dir / f"{prefix}_slices.png")
    plt.close(fig)

    # Save the mesh
    if args.mesh:
        threshold = 0.5
        print("Saving predicted mesh")
        _save_mesh(prediction, prefix, threshold, out_dir)

        # Save the mesh on top of the ground truth
        if args.test:
            print("Saving test meshes")
            thresholded = prediction > threshold
            _save_test_meshes(thresholded, truth, out_dir)


def _inference(args: argparse.Namespace, net: torch.nn.Module, config: dict) -> None:
    """
    Do the inference, save the plots

    """
    # Find which activation function to use from the config file
    # This assumes this was the same activation function used during training...
    activation = model.activation_name(config)

    # Either iterate over all subjects or just do the one
    if args.all:
        for subject in read.crop_lookup().keys():
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

    # If we're doing the splotch thing, we need to have an rng
    if args.splotch:
        args.rng = np.random.default_rng(seed=0)

    # Load the model and training-time config
    print("Loading model")
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
        choices=set(read.crop_lookup().keys()),
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--speckle",
        help="Make the segmentation worse by removing every second predicted voxel"
        "(to illustrate the effect on our metrics)",
        action="store_true",
    )
    group.add_argument(
        "--splotch",
        help="Make the segmentation worse by adding random blobs of noise to the prediction"
        "(to illustrate the effect on our metrics)",
        action="store_true",
    )

    main(parser.parse_args())
