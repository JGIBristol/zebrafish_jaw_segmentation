"""
Find the location to crop from in the rear jaw dataset.

This script is how I found the locations in data/jaw_centres.csv
for the rear jaw dataset.

"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from fishjaw.images import io
from fishjaw.images import transform
from fishjaw.util import util


def find_xy(
    binary_img: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Find the point in the (2d) image in the centre of the white pixels

    """
    white_pixel_coords = np.argwhere(binary_img == 1)

    # Calculate the center of mass
    if white_pixel_coords.size == 0:
        raise ValueError("The binary image contains no white pixels.")

    center_of_mass = white_pixel_coords.mean(axis=0)

    return tuple(center_of_mass)


def main():
    """
    Read the DICOMs, find the last slice that contains labels, then find the XY location of the labels

    """
    plot_dir = ...  # files.script_dir() or something
    if not plot_dir.is_dir():
        plot_dir.mkdir()

        folder = pathlib.Path("dicoms/Training set 3 (base of jaw)")
        for path in folder.glob("*.dcm"):
            # Open each DICOM in the training set 3 folder
            _, mask = io.read_dicom(path)

            # Find the last Z slice for which there are at least some nonzero values
            n_required = 3
            z_nonzero = np.sum(mask, axis=(1, 2)) > n_required
            idx = len(z_nonzero) - np.argmax(z_nonzero[::-1])

            # Get the size of the crop window from the config
            config = util.userconf()
            crop_size = transform.window_size(config)

            # If our crop window would overlap with the edge of the image in Z, error
            # We don't want to fiddle with the Z location since this might
            # accidentally make the window contain unlabelled jaw
            if transforms.crop_out_of_bounds(*transforms.start_and_end(idx, crop_size[0], start_from_loc=True), mask.shape[0]):
                raise ValueError(f"Z crop out of bounds for {path}: {transforms.start_and_end(idx, crop_size[0], start_from_loc=True)} bound for image size {mask.shape[0]}"))
            # Error

            # Find the xy centre of that slice
            x, y = find_xy(mask[idx - 1])

            # Check if this overlaps with the edge of the image
            # Warn
            # Move the XY location such that it doesn't overlap

            # Find n from the path
            n = int(path.stem.split("_", maxsplit=1)[-1])

            # Plot them
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i, (axis, img) in enumerate(zip(axes, mask[idx - 2 : idx + 1])):
                axis.imshow(img)
                axis.plot(y, x, "ro")
            fig.savefig(plot_dir / f"{n}.png")
            plt.close(fig)

            # Output n,z
            print(n, idx, round(x), round(y), "FALSE", sep=",")


if __name__ == "__main__":
    main()
