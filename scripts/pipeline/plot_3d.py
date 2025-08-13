"""
Plot projections of the jaw in 3D with the greyscale value of voxels preserved
"""

import argparse
import tifffile

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from fishjaw.util import files
from fishjaw.visualisation import images_3d


def _calculate_point_size(axis: plt.Axes, img_size: tuple[int, int, int]) -> float:
    """Calculate point size so each point is roughly 1 pixel"""
    assert set(img_size) == {img_size[0]}, "Image must be cubic"

    fig_width_px = (
        axis.figure.get_figwidth() * axis.figure.dpi * axis.get_position().width
    )
    return (fig_width_px / img_size[0]) ** 2


def main():
    """
    Read in pairs of images and masks, plot projections and save them
    """
    in_dir = files.script_out_dir() / "jaw_segmentations"
    img_in_dir = in_dir / "imgs"
    mask_in_dir = in_dir / "masks"

    out_dir = in_dir / "projections"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = sorted(list(img_in_dir.glob("*.tif")))
    in_masks = sorted(list(mask_in_dir.glob("*.tif")))

    plot_kw = {"marker": "s", "cmap": "inferno", "vmin": 0, "vmax": 2**15}
    for img, mask in tqdm(zip(in_imgs, in_masks, strict=True), total=len(in_imgs)):
        i = tifffile.imread(img)
        m = tifffile.imread(mask)

        fig, (ax1, ax2) = plt.subplots(
            1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5)
        )

        if "s" not in plot_kw:
            # Calculate point size so each point is roughly 1 pixel
            # Assumes all the images are the same size, which they probably
            # are
            img_size = tifffile.imread(img).shape
            plot_kw["s"] = _calculate_point_size(plt.gca(), img_size)

        co_ords = np.argwhere(m)
        greyscale_vals = i[co_ords[:, 0], co_ords[:, 1], co_ords[:, 2]]

        for a in (ax1, ax2):
            scatter = a.scatter(
                co_ords[:, 0], co_ords[:, 1], co_ords[:, 2], **plot_kw, c=greyscale_vals
            )
            a.axis("off")

        fig.colorbar(scatter, ax=[ax1, ax2], shrink=0.5, aspect=20)

        ax1.view_init(elev=45, azim=-90, roll=-140)
        ax2.view_init(elev=180, azim=30)

        fig.savefig(out_dir / img.name.replace(".tif", ".png"))
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    main(**vars(args))
