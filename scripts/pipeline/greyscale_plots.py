"""
Simple analysis of the greyscale content of the segmentations.

This is interesting because it tells us about the bone density at each voxel.

This script will:
 - Plot histograms of the greyscale distribution of each jaw
"""

import pathlib
import argparse
import tifffile

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files


def _hist(vals: np.ndarray, path: pathlib.Path) -> None:
    """
    Plot a hist of greyscale vals and save it to the specified path
    """
    fig, axis = plt.subplots(figsize=(8, 6))

    axis.hist(vals, bins=np.linspace(0, 2**16, 100), histtype="stepfilled")

    axis.set_ylim(0, max(2500, axis.get_ylim()[1]))
    axis.set_title(path.stem)

    fig.tight_layout()
    fig.savefig(path)

    plt.close(fig)


def _process_pair(img: pathlib.Path, mask: pathlib.Path, hist_out_dir: pathlib.Path):
    """
    Read the images + masks, get the greyscale pixels and plot a histogram

    """
    hist_out_path = hist_out_dir / img.name.replace(".tif", ".png")
    if hist_out_path.is_file():
        return f"Skipping {hist_out_path.stem}"

    i = tifffile.imread(img)
    m = tifffile.imread(mask)

    greyscale_vals = i[m]

    # Plot and save a histogram
    _hist(greyscale_vals, hist_out_path)
    return f"Done {hist_out_path.stem}"


def main():
    """
    Read in the mastersheet to get metadata from the different segmentations

    Then read in pairs of images and masks, and extract the jaw voxels for each segmentation.

    Then make the plots - 1d histograms of each.
    """
    in_dir = files.script_out_dir() / "jaw_segmentations"
    img_in_dir = in_dir / "imgs"
    mask_in_dir = in_dir / "masks"

    hist_out_dir = in_dir / "hists"
    hist_out_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = sorted(list(img_in_dir.glob("*.tif")))
    in_masks = sorted(list(mask_in_dir.glob("*.tif")))

    for img, mask in tqdm(zip(in_imgs, in_masks, strict=True), total=len(in_imgs)):
        hist_out_path = hist_out_dir / img.name.replace(".tif", ".png")

        i = tifffile.imread(img)
        m = tifffile.imread(mask)

        greyscale_vals = i[m]

        # Plot and save a histogram
        _hist(greyscale_vals, hist_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()

    main(**vars(args))
