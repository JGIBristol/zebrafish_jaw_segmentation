"""
Simple analysis of the segmentations' shape.

This script just plots the age vs volume of the segmentation mask,
excluding segmentations that are obviously broken (e.g. contrast enhanced,
or where we've failed to find the jaw centre)

"""

import pathlib
import argparse
import tifffile

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.inference import read


def main():
    """
    Read in the mastersheet to get metadata from the different segmentations, so
    that we can get their ages and exclude the contrast enhanced fish. Exclude
    also those which are broken, but include the training data.

    Then read in the masks, get their volume, and make the plot of age vs volume.
    """
    in_dir = files.script_out_dir() / "jaw_segmentations"
    mask_in_dir = in_dir / "masks"
    in_masks = sorted(list(mask_in_dir.glob("*.tif")))

    out_path = in_dir / "volume_age_plot.png"

    ages = []
    volumes = []

    for mask_path in tqdm(in_masks, total=len(in_masks)):
        fish_n = read.fish_number(mask_path)
        metadata: read.Metadata = read.metadata(fish_n)

        # Check the metadata for inclusion - we might want to skip it, in which case print
        if read.is_excluded(fish_n, exclude_train_data=False, exclude_unknown_age=True):
            continue

        # Calculate the volume of the mask
        m = tifffile.imread(mask_path)
        vol = np.sum(m) * metadata.voxel_volume

        ages.append(metadata.age)
        volumes.append(vol)

    # Perform a simple linear fit to get the trendline
    z, cov = np.polyfit(ages, volumes, 1, cov=True)
    p = np.poly1d(z)

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(ages, volumes, alpha=0.5)
    axis.set_xlabel("Age (months)")
    axis.set_ylabel(r"Volume $\left(mm^3\right)$")

    unique_ages = list(set(ages))
    axis.plot(unique_ages, p(unique_ages), color="red", linestyle="--")

    axis.set_title(
        f"Average increase: {z[0]:.3f}$\pm${np.sqrt(cov[0, 0]):.3f} $mm^3$/month\nN={len(ages)}"
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()

    main(**vars(args))
