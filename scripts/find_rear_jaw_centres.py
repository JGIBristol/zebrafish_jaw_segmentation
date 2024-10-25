"""
Find the location to crop from in the rear jaw dataset.

This script is how I found the locations in data/jaw_centres.csv
for the rear jaw dataset.

"""
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from fishjaw.images import io


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
    plot_dir = pathlib.Path("tmp/")
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

            # Find the xy centre of that slice
            x, y = find_xy(mask[idx - 1])

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
