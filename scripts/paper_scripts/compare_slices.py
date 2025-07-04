"""
Compare slices through the segmentation between humans and the model,
and plot some projections of the segmentation in 3D.

"""

import pathlib
import argparse

import tifffile
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.gridspec import GridSpec

from fishjaw.util import files, util
from fishjaw.model import model, data
from fishjaw.images import metrics, transform
from fishjaw.inference import read


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


def _plot_slices(
    out_dir: pathlib.Path,
    scan: NDArray,
    felix: NDArray,
    harry: NDArray,
    tahlia: NDArray,
    inference: NDArray,
) -> None:
    """
    Plot slices on a figure and save it
    """
    fig, axes = plt.subplots(2, 2)
    n = 69
    vmin, vmax = np.min(scan[n]), np.max(scan[n])
    for name, label, axis in zip(
        ["felix", "harry", "tahlia", "inference"],
        ["P1", "P2", "P3", "Inference"],
        axes.flat,
    ):
        axis.imshow(scan[n], cmap="gray", vmin=vmin, vmax=vmax)
        axis.imshow(locals()[name][n], cmap="hot_r", alpha=0.5)
        axis.set_title(label)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "compare_slices.png")


def _plot_projections(
    out_dir: pathlib.Path,
    felix: NDArray,
    harry: NDArray,
    tahlia: NDArray,
    inference: NDArray,
) -> None:
    """
    Plot projection on a figure and save it

    """
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2, figure=fig, hspace=0.05, wspace=0.05)

    side_axes = [fig.add_subplot(int(f"42{i}"), projection="3d") for i in "1256"]
    top_axes = [fig.add_subplot(int(f"42{i}"), projection="3d") for i in "3478"]

    pbar = tqdm.tqdm(total=8, desc="Plotting projections")
    plot_kw = {"cmap": "copper", "alpha": 1, "s": 2, "marker": "s"}
    for i, x in enumerate((felix, harry, tahlia, inference)):
        co_ords = np.argwhere(x > 0.5)
        side_axis, top_axis = (side_axes[i], top_axes[i])
        side_axis.scatter(
            co_ords[:, 0], co_ords[:, 1], co_ords[:, 2], c=co_ords[:, 2], **plot_kw
        )
        side_axis.view_init(elev=45, azim=-90, roll=-140)
        pbar.update(1)

        top_axis.scatter(
            co_ords[:, 0], co_ords[:, 1], co_ords[:, 2], c=co_ords[:, 2], **plot_kw
        )
        top_axis.view_init(azim=30, elev=180)
        pbar.update(1)

    # for axis in side_axes + top_axes:
    #     axis.axis("off")

    fig.tight_layout()

    fig.savefig(out_dir / "compare_projections.png")


def main(model_name: str) -> None:
    """
    Read images from the RDSF and the model from disk, perform inference
    then plot slices
    """
    config = util.userconf()

    out_dir = files.script_out_dir() / "compare_slices"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # perform inference
    print("Performing inference")
    inference = _inference(model_name)

    # load human segmentations
    print("Loading human segmentations")
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

    felix, harry, tahlia = (
        transform.crop(
            x, read.crop_lookup()[97], transform.window_size(config), centred=True
        )
        for x in (felix, harry, tahlia)
    )

    # Read the original image
    scan = read.cropped_img(config, 97)

    _plot_slices(out_dir, scan, felix, harry, tahlia, inference)
    _plot_projections(out_dir, felix, harry, tahlia, inference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a selected slice through the segmentation between humans and the model"
    )
    parser.add_argument(
        "--model_name",
        help="Which model to load from the models dir; e.g. 'model_state.pkl'",
        default="paper_model.pkl",
    )

    main(**vars(parser.parse_args()))
