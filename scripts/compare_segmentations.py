"""
Compare some human segmentations to each other and to a model's segmentation

"""

import pathlib
import argparse

import tifffile
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.model import model, data
from fishjaw.images import metrics, transform
from fishjaw.visualisation import plot_meshes as mesh_lib
from fishjaw.inference import read, mesh
from fishjaw.util import util


def plot_meshes(
    thresholded_pred: np.ndarray,
    baseline: np.ndarray,
    out_dir: pathlib.Path,
    label: str,
) -> None:
    """
    Turn a segmentation into a mesh and save it

    """
    # Create meshes for plotting
    prediction_mesh = mesh.cubic_mesh(thresholded_pred)
    truth_mesh = mesh.cubic_mesh(baseline)

    # Create the figure
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))

    # Find the Hausdorff points
    hausdorff_points = metrics.hausdorff_points(baseline, thresholded_pred)

    # Make projections of the meshes
    mesh_lib.projections(
        axes,
        prediction_mesh,
        plot_kw={"alpha": 0.2, "color": "blue", "label": label},
    )
    mesh_lib.projections(
        axes, truth_mesh, plot_kw={"alpha": 0.1, "color": "grey", "label": "Felix"}
    )

    # Indicate Hausdorff distance
    x, y, z = zip(*hausdorff_points)
    for ax in axes:
        ax.plot(x, y, z, "rx-", markersize=4, label="Hausdorff distance")

    axes[0].legend(loc="upper right")

    fig.savefig(
        f"{out_dir}/overlay_{label}.png",
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

    prediction = model.predict(
        net,
        read.inference_subject(config, 97),
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=model.activation_name(config),
    )

    # Threshold the segmentation
    prediction = (prediction > 0.5).astype(np.uint8)

    return metrics.largest_connected_component(prediction)


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

    # Load the human segmentations
    print("loading human segmentations")
    config = util.userconf()
    seg_dir = (
        files.rdsf_dir(config)
        / "1Felix and Rich make models"
        / "Human validation STL and results"
    )
    felix = tifffile.imread(
        seg_dir / "felix take2" / "ak_97-FBowers_complete.labels.tif"
    )
    harry = tifffile.imread(seg_dir / "Harry" / "ak_97.tif.labels.tif")
    tahlia = tifffile.imread(seg_dir / "Tahlia" / "tpollock_97_avizo.labels.tif")

    # Perform inference with the model
    print("Performing inference")
    inference = _inference(model_name)

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

    # Save it as a mesh
    print("Saving the inference as a mesh")
    mesh.cubic_mesh(inference).save(out_dir / "inference.stl")

    # Modify the model segmentation to make it worse
    print("Modifying the model segmentation")
    speckled = transform.speckle(inference)
    splotched = transform.add_random_blobs(rng, inference)

    # Compare the segmentations to a baseline, print a table of metrics
    print("Comparing segmentations")
    segmentations = [felix, harry, tahlia, inference, speckled, splotched]
    table = metrics.table(
        [felix] * 6,
        segmentations,
        thresholded_metrics=True,
    )
    table["label"] = ["felix", "harry", "tahlia", "inference", "speckled", "splotched"]
    table.set_index("label", inplace=True)
    print(table.to_markdown())

    for label, segmentation in zip(table.index, segmentations):
        plot_meshes(segmentation, felix, out_dir, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )
    parser.add_argument(
        "model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
    )
    main(**vars(parser.parse_args()))
