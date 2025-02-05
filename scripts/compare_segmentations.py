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
from fishjaw.util import util


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
    print("Performing inference")
    inference = _inference(model_name)

    # Load the human segmentations
    print("loading human segmentations")
    config = util.userconf()
    seg_dir = (
        files.rdsf_dir(config)
        / "1Felix and Rich make models"
        / "Human validation STL and results"
    )
    felix = tifffile.imread(seg_dir / "Felix" / "ak_97.tif.labels(2).tif")
    harry = tifffile.imread(seg_dir / "Harry" / "ak_97.tif.labels.tif")
    tahlia = tifffile.imread(seg_dir / "Tahlia" / "tpollock_97_avizo.labels.tif")

    # Crop them to the same size as the model's output
    print("Cropping")
    felix, harry, tahlia = (
        transform.crop(
            x, read.crop_lookup()[97], transform.window_size(config), centred=True
        )
        for x in (felix, harry, tahlia)
    )

    # Save the inference as a tiff
    print("Saving the inference")
    tifffile.imwrite(out_dir / "inference.tif", inference)

    # Modify the model segmentation to make it worse
    print("Modifying the model segmentation")
    speckled = transform.speckle(inference)
    splotched = transform.add_random_blobs(rng, inference)
    tifffile.imwrite(out_dir / "speckled.tif", speckled)
    tifffile.imwrite(out_dir / "splotched.tif", splotched)

    # Compare the segmentations to a baseline, print a table of metrics
    print("Comparing segmentations")
    table = metrics.table(
        [felix] * 6,
        [felix, harry, tahlia, inference, speckled, splotched],
        thresholded_metrics=True,
    )
    table["label"] = ["felix", "harry", "tahlia", "inference", "speckled", "splotched"]
    table.set_index("label", inplace=True)
    print(table.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )
    parser.add_argument(
        "model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
    )
    main(**vars(parser.parse_args()))
