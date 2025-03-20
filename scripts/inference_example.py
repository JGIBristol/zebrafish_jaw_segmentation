"""
Perform inference on an out-of-sample subject

"""

import pathlib
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import tifffile
import numpy as np
import torchio as tio
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.model import model, data
from fishjaw.images import metrics, transform
from fishjaw.visualisation import images_3d, plot_meshes
from fishjaw.inference import read, mesh


def rotating_plots(mask: np.ndarray, out_dir: pathlib.Path, img_n: int) -> None:
    """
    Save an lots of images of a rotating mesh, which we can then
    turn into a gif

    You can turn these into an mp4 with e.g.
    ffmpeg -framerate 12 -pattern_type glob -i 'script_output/inference/new_jaws/rotating_mesh/317/*.png' -c:v libx264 317.mp4

    """
    plt.switch_backend("agg")
    plt_lock = threading.Lock()

    plot_dir = out_dir / "rotating_mesh" / f"{img_n}"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    def plot_proj(enum_angles):
        """Helper fcn to change the rotation of a plot"""
        i, angles = enum_angles
        axis.view_init(*angles)
        with plt_lock:
            fig.savefig(f"{plot_dir}/mesh_{i:03}.png")
            plt.close(fig)
        return i

    # Make a scatter plot of the mask
    fig = plt.figure()
    axis = fig.add_subplot(projection="3d")
    co_ords = np.argwhere(mask > 0.5)
    axis.scatter(
        co_ords[:, 0],
        co_ords[:, 1],
        co_ords[:, 2],
        c=co_ords[:, 2],
        cmap="copper",
        alpha=0.5,
    )
    axis.axis("off")

    # Plot the mesh at various angles
    num_frames = 108
    azimuths = np.linspace(-90, 270, num_frames, endpoint=False)
    elevations = list(np.linspace(-90, 90, num_frames // 2)) + list(
        np.linspace(90, -90, num_frames // 2)
    )
    rolls = np.linspace(0, 360, num_frames, endpoint=False)

    angles = list(enumerate(zip(azimuths, elevations, rolls)))

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(plot_proj, angle) for angle in angles]
        for _ in tqdm(as_completed(futures), total=len(angles)):
            pass


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
        plot_kw={"alpha": 0.3, "color": "blue", "label": "Prediction"},
    )
    plot_meshes.projections(
        axes, truth_mesh, plot_kw={"alpha": 0.2, "color": "red", "label": "Truth"}
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

    if args.plot_angles:
        rotating_plots(prediction, out_dir, args.subject)

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
        for subject in read.crop_lookup():
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
        "--plot_angles",
        help="Plot lots of pngs of the segmentation at various angles",
        action="store_true",
    )
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
