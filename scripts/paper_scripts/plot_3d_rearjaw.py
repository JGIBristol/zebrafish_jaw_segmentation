"""
Plot some projections of a 3d rear-jaw-only training image

"""

import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from fishjaw.util import files, util
from fishjaw.model import data
from fishjaw.images import transform


def _calculate_point_size(axis: plt.Axes, img_size: tuple[int, int, int]) -> float:
    """Calculate point size so each point is roughly 1 pixel"""
    assert set(img_size) == {img_size[0]}, "Image must be cubic"

    fig_width_px = (
        axis.figure.get_figwidth() * axis.figure.dpi * axis.get_position().width
    )
    return (fig_width_px / img_size[0]) ** 2


def _plot_projections(out_dir: pathlib.Path, image: NDArray) -> None:
    """
    Plot projection on a figure and save it

    """
    fig = plt.figure(figsize=(10, 10))
    side_axis = fig.add_subplot(121, projection="3d")
    top_axis = fig.add_subplot(122, projection="3d")

    plot_kw = {
        "cmap": "copper",
        "alpha": 1,
        "marker": "s",
        "s": _calculate_point_size(side_axis, image.shape),
    }
    co_ords = np.argwhere(image)

    side_axis.scatter(
        co_ords[:, 0],
        co_ords[:, 1],
        co_ords[:, 2],
        c=co_ords[:, 2],
        **plot_kw,
    )
    side_axis.view_init(elev=45, azim=-90, roll=-140)

    top_axis.scatter(
        co_ords[:, 0],
        co_ords[:, 1],
        co_ords[:, 2],
        c=co_ords[:, 2],
        **plot_kw,
    )
    top_axis.view_init(azim=30, elev=180)

    for axis in (side_axis, top_axis):
        axis.axis("off")

    fig.savefig(out_dir / "rear_jaw_3d.png", transparent=True)


def main() -> None:
    """
    Read images from the RDSF and the model from disk, perform inference
    then plot slices
    """
    # We read files from the folders as specified in the config.
    # Therefore, we'll fiddle with the config so that we only read from the
    # folders containing the rear jaws
    config = util.userconf()
    config["dicom_dirs"] = [config["dicom_dirs"][1]]
    assert (
        "base of jaw" in config["dicom_dirs"][0]
    ), "Expected dicom_dirs in userconf.yml to have the base of jaws dir in the second position"

    out_dir = files.script_out_dir() / "3d_projection"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # load an example image
    dicom_path, *_ = files.dicom_paths(config, mode="train")
    # only take the labels for plotting
    _, scan = data.cropped_dicom(dicom_path, transform.window_size(config))

    # plot it
    _plot_projections(out_dir, scan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(**vars(parser.parse_args()))
