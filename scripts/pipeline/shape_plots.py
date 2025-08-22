"""
Simple analysis of the segmentations' shape.

This script plots the age vs volume of the segmentation mask,
excluding segmentations that are obviously broken (e.g. contrast enhanced,
or where we've failed to find the jaw centre)

It also plots pairs of age/volume/length, since we might expect a bigger (longer)
fish to have a bigger jaw

"""

import pathlib
import argparse
import tifffile

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.inference import read


def _all_plots(
    ages: list[int],
    volumes: list[float],
    lengths: list[float],
    genotypes: list[str],
    out_path: pathlib.Path,
):
    """
    Create all plots for the given ages, volumes, and lengths.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    ages = np.array(ages)
    volumes = np.array(volumes)
    lengths = np.array(lengths)
    genotypes = np.array(genotypes)

    # Colour code based on genotype
    unique_genotypes = np.unique(genotypes)
    for g in unique_genotypes:
        a = ages[genotypes == g]
        v = volumes[genotypes == g]
        l = lengths[genotypes == g]

        axes[0].scatter(a, v, alpha=0.5)
        axes[0].set_xlabel("Age")
        axes[0].set_ylabel("Vol")

        axes[1].scatter(a, l, alpha=0.5)
        axes[1].set_xlabel("Age")
        axes[1].set_ylabel("Length")

        axes[2].scatter(v, l, alpha=0.5, label=g)
        axes[2].set_xlabel("Vol")
        axes[2].set_ylabel("Length")

    axes[2].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_vol_vs_age(ages: list[int], volumes: list[float], out_path: pathlib.Path):
    """
    Plot volume vs age
    """
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
    lengths = []
    genotypes = []

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
        lengths.append(metadata.length)
        genotypes.append(metadata.genotype)

    _plot_vol_vs_age(ages, volumes, out_path)
    _all_plots(ages, volumes, lengths, genotypes, out_path.with_name("all_plots.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()

    main(**vars(args))
