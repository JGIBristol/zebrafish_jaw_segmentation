"""
Compare some human segmentations to each other and to a model's segmentation

"""

import pathlib
import argparse

import torch
import tifffile
import numpy as np
from numpy.typing import NDArray
import torchio as tio
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.model import model, data
from fishjaw.images import metrics, transform
from fishjaw.visualisation import images_3d, plot_meshes
from fishjaw.inference import read, mesh


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

    # Remove every second voxel to make the segmentation worse
    if args.speckle:
        prediction[::2, ::2, ::2] = 0
        prediction[1::2, 1::2, ::2] = 0
        prediction[::2, 1::2, 1::2] = 0
        prediction[1::2, ::2, 1::2] = 0

    if args.splotch:
        prediction = _add_random_blobs(args.rng, prediction)

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


def _inference(model_name: str) -> NDArray:
    """
    Do the inference, save the plots

    """
    # Load the model and training-time config
    model_state = model.load_model(model_name)

    config = model_state.config
    net = model_state.load_model(set_eval=True)
    net.to("cuda")

    # Threshold the segmentation
    return (
        model.predict(
            net,
            read.inference_subject(config, 97),
            patch_size=data.get_patch_size(config),
            patch_overlap=(4, 4, 4),
            activation=model.activation_name(config),
            batch_size=4,
        )
        > 0.5
    ).astype(np.uint8)


def main(*, model_name: str):
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    rng = np.random.default_rng(seed=0)

    out_dir, _ = model_name.split(".pkl")
    assert not _, "Model name should end with .pkl"
    out_dir = files.script_out_dir() / "comparison" / out_dir
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # Perform inference with the model
    inference = _inference(model_name)

    # Load the human segmentations

    # Save the inference as a tiff
    tifffile.imwrite(out_dir / "inference.tif", inference)

    # Modify the model segmentation to make it worse
    speckled = transform.speckle(inference, rng)
    splotched = transform.add_random_blobs(rng, speckled)

    # Compare the segmentations to a baseline, print a table of metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )
    parser.add_argument(
        "model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
    )
    main(**vars(parser.parse_args()))
