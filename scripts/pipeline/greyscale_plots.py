"""
Simple analysis of the greyscale content of the segmentations.

This is interesting because it tells us about the bone density at each voxel.

This script will:
 - Plot histograms of the greyscale distribution of each jaw
 - Plot boxplots of the greyscale distribution of the jaws, grouped to have
   10 per plot. This part of the script is just a quick prototype, it
   currently isn't very useful for presentation of analysis
"""

import pathlib
import argparse
import tifffile

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files


def _create_boxplot(data, labels, out_dir, plot_number):
    """Create and save a boxplot for the current batch"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(data, tick_labels=labels, patch_artist=True)

    ax.set_title("Greyscale Values")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 2**16)

    fig.tight_layout()

    output_path = out_dir / f"boxplot_{plot_number}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _hist(vals: np.ndarray, path: pathlib.Path) -> None:
    """
    Plot a hist of greyscale vals and save it to the specified path
    """
    fig, axis = plt.subplots(figsize=(8, 6))

    axis.hist(vals, bins=np.linspace(0, 2**15, 100), histtype="stepfilled")
    axis.set_title(path.stem)

    fig.tight_layout()
    fig.savefig(path)

    plt.close(fig)


def main(boxplot: bool):
    """
    Read in the mastersheet to get metadata from the different segmentations

    Then read in pairs of images and masks
    """
    mastersheet = files._mastersheet()
    mastersheet.set_index("n")

    in_dir = files.script_out_dir() / "jaw_segmentations"
    img_in_dir = in_dir / "imgs"
    mask_in_dir = in_dir / "masks"

    out_dir = in_dir / "boxplot"
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_out_dir = in_dir / "hists"
    hist_out_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = sorted(list(img_in_dir.glob("*.tif")))
    in_masks = sorted(list(mask_in_dir.glob("*.tif")))

    # Storage for current batch of boxplots
    batch_data = []
    batch_labels = []
    batch_count = 0
    plot_number = 1

    for img, mask in tqdm(zip(in_imgs, in_masks, strict=True), total=len(in_imgs)):
        i = tifffile.imread(img)
        m = tifffile.imread(mask)

        greyscale_vals = i[m]

        # Plot and save a histogram
        _hist(greyscale_vals, hist_out_dir / img.name.replace(".tif", ".png"))

        # For speed i've added an option to skip the boxplot, which is relatively slow
        if not boxplot:
            continue

        # Add a boxplot to the current batch
        batch_data.append(greyscale_vals)
        batch_labels.append(img.stem)
        batch_count += 1

        # Create plot every 10 iterations
        if batch_count == 10:
            _create_boxplot(batch_data, batch_labels, out_dir, plot_number)

            # Reset for next batch
            batch_data = []
            batch_labels = []
            batch_count = 0
            plot_number += 1

    # Handle remaining data (if not exactly divisible by 10)
    if batch_data:
        _create_boxplot(batch_data, batch_labels, out_dir, plot_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxplot", action="store_true", help="Also make boxplots")

    args = parser.parse_args()

    main(**vars(args))
