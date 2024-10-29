"""
Find the location to crop from in the rear jaw dataset.

This script is how I found the locations in data/jaw_centres.csv
for the rear jaw dataset.

"""

import pathlib
import warnings

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
    Read the DICOMs, find the last slice that contains labels,
    then find the XY location of the labels

    """

    # We will be printing our results to terminal- but we want to keep track of
    # if/where I've fiddled with the X and Y locations to avoid a partial crop
    # So we'll record a log of warnings to output as well as the results
    warning_buffer = []
    results_buffer = []

    plot_dir = pathlib.Path(util.config()["script_output"]) / "find_rear_jaw_centres"
    if not plot_dir.is_dir():
        plot_dir.mkdir()

    # Only look at the training set 3 folder
    folder = pathlib.Path("dicoms/Training set 3 (base of jaw)")
    assert folder.is_dir()

    # Get the size of the crop window from the config
    config = util.userconf()
    crop_size = transform.window_size(config)
    for path in folder.glob("*.dcm"):
        # Open each DICOM in the training set 3 folder
        _, mask = io.read_dicom(path)

        # Find the last Z slice for which there are at least some nonzero values
        n_required = 3
        z_nonzero = np.sum(mask, axis=(1, 2)) > n_required
        idx = len(z_nonzero) - np.argmax(z_nonzero[::-1])

        # If our crop window would overlap with the edge of the image in Z, error
        # We don't want to fiddle with the Z location since this might
        # accidentally make the window contain unlabelled jaw
        if transform.crop_out_of_bounds(
            *transform.start_and_end(idx, crop_size[0], start_from_loc=True),
            mask.shape[0],
        ):
            raise ValueError(
                f"""Z crop out of bounds for {path}:
                    {transform.start_and_end(idx,
                                             crop_size[0],
                                             start_from_loc=True,
                                             )}
                    bound for image size {mask.shape[0]}
                 """
            )

        # Find the xy centre of that slice
        x, y = find_xy(mask[idx - 1])

        # Check if this overlaps with the edge of the image
        # start_from_loc is False here since we crop centrally around X and Y
        if transform.crop_out_of_bounds(
            *(bounds := transform.start_and_end(x, crop_size[1], start_from_loc=False)),
            mask.shape[1],
        ):
            warning_buffer.append(
                f"""X crop out of bounds for {path}:
                    {bounds=}
                    for image size {mask.shape[1]}
                 """
            )
            print(bounds)
            # Move the X location such that it doesn't overlap with the edge
            # x = mask.shape[1]

        if transform.crop_out_of_bounds(
            *(bounds := transform.start_and_end(y, crop_size[2], start_from_loc=False)),
            mask.shape[2],
        ):
            warning_buffer.append(
                f"""Y crop out of bounds for {path}:
                    {bounds=}
                    for image size {mask.shape[2]}
                 """
            )
            print(bounds)
            # Move the X location such that it doesn't overlap with the edge
            # y = mask.shape[2]

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
        # This is what we'll copy and paste into the csv
        results_buffer.append(f"{n},{idx},{round(x)},{round(y)},FALSE")

    # Print the results
    print("\n".join(results_buffer))

    for message in warning_buffer:
        warnings.warn(message)


if __name__ == "__main__":
    main()
