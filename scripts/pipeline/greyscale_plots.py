"""
Simple analysis of the greyscale content of the segmentations.

This is interesting because it tells us about the bone density at each voxel.

This script will:
 - Plot histograms of the greyscale distribution of each jaw
 - Plot boxplots of the greyscale distribution of the jaws, grouped to have
   10 per plot. This part of the script is just a quick prototype, it
   currently isn't very useful for presentation of analysis
"""

import argparse
import tifffile

from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.visualisation import images_3d


def main():
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

    in_imgs = sorted(list(img_in_dir.glob("*.tif")))
    in_masks = sorted(list(mask_in_dir.glob("*.tif")))

    # Storage for current batch
    batch_data = []
    batch_labels = []
    batch_count = 0
    plot_number = 1

    for img, mask in tqdm(zip(in_imgs, in_masks, strict=True), total=len(in_imgs)):
        i = tifffile.imread(img)
        m = tifffile.imread(mask)

        greyscale_vals = i[m]

        # Add to current batch
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


def _create_boxplot(data, labels, out_dir, plot_number):
    """Create and save a boxplot for the current batch"""
    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    # Customize appearance
    ax.set_title(f"Greyscale Values")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 2**16)

    fig.tight_layout()

    output_path = out_dir / f"boxplot_{plot_number}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    main(**vars(args))
