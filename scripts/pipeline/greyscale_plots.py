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


def _lineplot(
    x: np.ndarray,
    averages: tuple[np.ndarray, np.ndarray],
    quartiles: tuple[np.ndarray, np.ndarray],
    std: np.ndarray,
    label: str,
    path: pathlib.Path,
) -> None:
    """ """
    fig, axis = plt.subplots(figsize=(8, 6))

    # sort averages
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    averages = (averages[0][sort_indices], averages[1][sort_indices])
    quartiles = (quartiles[0][sort_indices], quartiles[1][sort_indices])
    std = std[sort_indices]

    median, mean = averages

    axis.plot(x, median, color="C0", label="Median")
    axis.fill_between(x, quartiles[0], quartiles[1], color="C0", alpha=0.2, label="IQR")

    axis.plot(x, mean, color="C1", label="Mean")
    axis.fill_between(x, mean - std, mean + std, color="C1", alpha=0.2, label="Std")

    axis.set_xlabel(label)
    axis.set_ylabel("Greyscale Intensity")
    axis.legend()

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
        _lineplot(
            data["length"],
            (data["median"], data["mean"]),
            (data["q25"], data["q75"]),
            data["std"],
            "Length (mm)",
            out_dir / "greyscale_vs_length.png",
        )

    if age:
        _lineplot(
            data["age"],
            (data["median"], data["mean"]),
            (data["q25"], data["q75"]),
            data["std"],
            "Age (months)",
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
