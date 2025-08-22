"""
Simple analysis of the greyscale content of the segmentations.

This is interesting because it tells us about the bone density at each voxel.

Use the command line flags to choose which kinds of plot to make
"""

import pathlib
import argparse
import tifffile

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.inference import read


def _ageplot(
    age: np.ndarray,
    averages: tuple[np.ndarray, np.ndarray],
    quartiles: tuple[np.ndarray, np.ndarray],
    std: np.ndarray,
    path: pathlib.Path,
):
    """
    Plot as points with errorbars
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)

    # sort averages
    sort_indices = np.argsort(age)
    age = age[sort_indices]
    averages = (averages[0][sort_indices], averages[1][sort_indices])
    quartiles = (quartiles[0][sort_indices], quartiles[1][sort_indices])
    std = std[sort_indices]

    median, mean = averages

    plot_kw = {"fmt": "o", "alpha": 0.5}
    axes[0].set_title("Median (IQR)")
    axes[0].errorbar(age, median, yerr=quartiles, color="C0", **plot_kw)

    axes[1].set_title("Mean ($\sigma$)")
    axes[1].errorbar(age, mean, yerr=std, color="C1", **plot_kw)

    axes[0].set_ylabel("Greyscale Intensity")
    for axis in axes:
        axis.set_xlabel("Age (months)")
        axis.set_ylim(0, 85000)

    fig.tight_layout()
    fig.savefig(path)

    plt.close(fig)


def _lenplot(
    length: np.ndarray,
    averages: tuple[np.ndarray, np.ndarray],
    quartiles: tuple[np.ndarray, np.ndarray],
    std: np.ndarray,
    path: pathlib.Path,
) -> None:
    """ """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # sort averages
    sort_indices = np.argsort(length)
    length = length[sort_indices]
    averages = (averages[0][sort_indices], averages[1][sort_indices])
    quartiles = (quartiles[0][sort_indices], quartiles[1][sort_indices])
    std = std[sort_indices]

    median, mean = averages

    axes[0].plot(length, median, color="C0")
    axes[0].fill_between(length, quartiles[0], quartiles[1], color="C0", alpha=0.2)

    axes[1].plot(length, mean, color="C1")
    axes[1].fill_between(length, mean - std, mean + std, color="C1", alpha=0.2)

    axes[0].set_ylabel("Greyscale Intensity")
    for axis in axes:
        axis.set_xlabel("Length (mm)")
        axis.set_ylim(0, 85000)

    axes[0].set_title("Median (IQR)")
    axes[1].set_title("Mean ($\sigma$)")

    fig.tight_layout()
    fig.savefig(path)

    plt.close(fig)


def _hist(vals: np.ndarray, path: pathlib.Path, metadata: read.Metadata) -> None:
    """
    Plot a hist of greyscale vals and save it to the specified path
    """
    fig, axis = plt.subplots(figsize=(8, 6))

    axis.hist(vals, bins=np.linspace(0, 2**16, 100), histtype="stepfilled")

    axis.set_ylim(0, max(4000, axis.get_ylim()[1]))
    axis.set_title(str(metadata))

    fig.tight_layout()
    fig.savefig(path)

    plt.close(fig)


def main(hists: bool, length: bool, age: bool):
    """
    Read in the mastersheet to get metadata from the different segmentations

    Then read in pairs of images and masks, and extract the jaw voxels for each segmentation.

    Then make the plots - 1d histograms of each, line plots of median/mean vs age
    """
    in_dir = files.script_out_dir() / "jaw_segmentations"
    img_in_dir = in_dir / "imgs"
    mask_in_dir = in_dir / "masks"

    out_dir = in_dir / "greyscale"
    out_dir.mkdir(parents=True, exist_ok=True)
    if hists:
        hist_dir = out_dir / "hists"
        hist_dir.mkdir(exist_ok=True)

    in_imgs = sorted(list(img_in_dir.glob("*.tif")))
    in_masks = sorted(list(mask_in_dir.glob("*.tif")))

    data = {
        k: np.empty(len(in_imgs))
        for k in ("length", "age", "median", "q25", "q75", "mean", "std")
    }

    for i, (img_path, mask_path) in tqdm(
        enumerate(zip(in_imgs, in_masks, strict=True)), total=len(in_imgs)
    ):
        # Get the metadata
        metadata: read.Metadata = read.metadata(read.fish_number(img_path))
        data["length"][i] = metadata.length
        data["age"][i] = metadata.age

        im = tifffile.imread(img_path)
        m = tifffile.imread(mask_path)

        greyscale_vals = im[m]

        if hists:
            # Plot and save histograms
            _hist(
                greyscale_vals,
                hist_dir / img_path.name.replace(".tif", ".png"),
                metadata,
            )

        if length or age:
            data["median"][i] = np.median(greyscale_vals)
            data["q25"][i] = np.percentile(greyscale_vals, 25)
            data["q75"][i] = np.percentile(greyscale_vals, 75)
            data["mean"][i] = np.mean(greyscale_vals)
            data["std"][i] = np.std(greyscale_vals)

    if length:
        _lenplot(
            data["length"],
            (data["median"], data["mean"]),
            (data["q25"], data["q75"]),
            data["std"],
            out_dir / "length.png",
        )

    if age:
        # Add a random jitter to age so that the plots look sensible
        data["age"] = data["age"] + np.random.uniform(-0.2, 0.2, size=len(data["age"]))
        _ageplot(
            data["age"],
            (data["median"], data["mean"]),
            (data["q25"], data["q75"]),
            data["std"],
            out_dir / "age.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hists", action="store_true", help="Plot 1d histograms of greyscale intensity"
    )
    parser.add_argument(
        "--length", action="store_true", help="Plot greyscale intensity vs length"
    )
    parser.add_argument(
        "--age", action="store_true", help="Plot greyscale intensity vs age"
    )

    args = parser.parse_args()

    main(**vars(args))
