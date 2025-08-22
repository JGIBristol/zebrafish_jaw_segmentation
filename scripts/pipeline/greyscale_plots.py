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

    Then make the plots - 1d histograms of each.
    """
    in_dir = files.script_out_dir() / "jaw_segmentations"
    img_in_dir = in_dir / "imgs"
    mask_in_dir = in_dir / "masks"

    hist_out_dir = in_dir / "hists"
    hist_out_dir.mkdir(parents=True, exist_ok=True)

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

        hist_out_path = hist_out_dir / img_path.name.replace(".tif", ".png")

        i = tifffile.imread(img_path)
        m = tifffile.imread(mask_path)

        greyscale_vals = i[m]

        if hists:
            # Plot and save histograms
            _hist(greyscale_vals, hist_out_path, metadata)

        if length or age:
            data["medians"][i] = np.median(greyscale_vals)
            data["q25"][i] = np.percentile(greyscale_vals, 25)
            data["q75"][i] = np.percentile(greyscale_vals, 75)
            data["means"][i] = np.mean(greyscale_vals)
            data["std"][i] = np.std(greyscale_vals)

    if length:
        ...

    if age:
        ...


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
