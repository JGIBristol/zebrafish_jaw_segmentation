"""
Compare a selected slice through the segmentation between humans and the model

"""

import argparse

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

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

    # plot slices
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
    fig.savefig(out_dir / f"compare_slices.png")


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
