"""
Find the location to crop from in the rear jaw dataset.

This script is how I found the locations in data/jaw_centres.csv
for the rear jaw dataset.

"""

import pathlib
import warnings

import tqdm
import numpy as np
import matplotlib.pyplot as plt

from fishjaw.images import io
from fishjaw.images import transform
from fishjaw.util import util


def _find_z_loc(mask: np.ndarray) -> int:
    """Find the last slice that contains labels"""
    n_required = 3
    z_nonzero = np.sum(mask, axis=(1, 2)) > n_required
    return len(z_nonzero) - np.argmax(z_nonzero[::-1])


def _find_xy(
    binary_img: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Find the point in the (3d) image in the centre of the white pixels

    """
    projection = binary_img.sum(axis=0)
    white_pixel_coords = np.argwhere(projection > 0)

    # Calculate the center of mass
    if white_pixel_coords.size == 0:
        raise ValueError("The binary image contains no white pixels.")

    weights = projection[projection > 0]
    center_of_mass = np.average(white_pixel_coords, axis=0, weights=weights)

    return tuple(center_of_mass)


def _check_z_overlap(
    mask: np.ndarray, idx: int, crop_size: tuple[int, int, int], path: pathlib.Path
):
    """
    Check if the crop window would overlap with the edge of the image in Z

    Error if so

    """
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


def _lateral_overlap(img_size: np.ndarray, x: int, crop_size: int) -> bool:
    """
    Check overlap in the X or Y directions
    """
    # start_from_loc is False here since we crop centrally around X and Y
    return transform.crop_out_of_bounds(
        *transform.start_and_end(x, crop_size, start_from_loc=False),
        img_size,
    )


def _shift_coords(co_ord: float, crop_size: int, max_length: int) -> float:
    """
    Shift the crop window if it would overlap with the edge of the image

    This check should be done before calling this function, otherwise
    something will probably go wrong

    """
    start, end = transform.start_and_end(co_ord, crop_size, start_from_loc=False)
    if start < 0:
        return co_ord - start
    elif end > max_length:
        return co_ord - (end - max_length)
    raise RuntimeError("Unexpected error")


def _find_excluded_pixels(
    mask: np.ndarray, bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
) -> np.ndarray:
    """
    Find the white pixels that have been cropped out of the mask

    """
    # bounds = [
    #     transform.start_and_end(a, b, start_from_loc=c)
    #     for (a, b, c) in zip(crop_coords, crop_size, [True, False, False])
    # ]

    # Find the co-ords of white pixels
    white_pxl_coords = np.argwhere(mask)

    # Find which ones are outside the crop window
    in_z_bounds = (white_pxl_coords[:, 0] >= bounds[0][0]) & (
        white_pxl_coords[:, 0] <= bounds[0][1]
    )
    in_y_bounds = (white_pxl_coords[:, 1] >= bounds[1][0]) & (
        white_pxl_coords[:, 1] <= bounds[1][1]
    )
    in_x_bounds = (white_pxl_coords[:, 2] >= bounds[2][0]) & (
        white_pxl_coords[:, 2] <= bounds[2][1]
    )

    return white_pxl_coords[~(in_z_bounds & in_y_bounds & in_x_bounds)]


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
    for path in tqdm.tqdm(
        folder.glob("*.dcm"),
        desc="Finding jaw centres",
        total=len(list(folder.glob("*.dcm"))),
    ):
        # Open each DICOM in the training set 3 folder
        _, mask = io.read_dicom(path)

        # Find the last Z slice for which there are at least some nonzero values
        idx = _find_z_loc(mask)

        # Find the xy centre of that slice
        x, y = _find_xy(mask)

        # If our crop window would overlap with the edge of the image in Z, error
        # We don't want to fiddle with the Z location since this might
        # accidentally make the window contain unlabelled jaw
        _check_z_overlap(mask, idx, crop_size, path)

        # Check if this overlaps with the edge of the image
        # This doesn't actually shift things if it's out of bounds, since this doesn't
        # happen often
        if _lateral_overlap(mask.shape[1], x, crop_size[1]):
            old_x = x
            x = _shift_coords(x, crop_size[1], mask.shape[1])
            warning_buffer.append(
                f"""X crop out of bounds for {path}:
                    image size {mask.shape}, {old_x=}, {crop_size[1]=}
                    Gives bounds {transform.start_and_end(x, crop_size[1], start_from_loc=False)}

                    X shifted to {x}
                 """
            )

        if _lateral_overlap(mask.shape[2], y, crop_size[2]):
            old_y = y
            y = _shift_coords(y, crop_size[2], mask.shape[2])
            warning_buffer.append(
                f"""Y crop out of bounds for {path}:
                    image size {mask.shape}, {old_y=}, {crop_size[2]=}
                    Gives bounds {transform.start_and_end(y, crop_size[2], start_from_loc=False)}

                    Y shifted to {y}
                 """
            )

        # Find n from the path
        n = int(path.stem.split("_", maxsplit=1)[-1])

        # Plot them
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, img in zip(axes, mask[idx - 2 : idx + 1]):
            axis.imshow(img)
            axis.plot(y, x, "ro")
        fig.savefig(plot_dir / f"{n}.png")
        plt.close(fig)

        # Output n,z
        # This is what we'll copy and paste into the csv
        crop_coords = idx, round(x), round(y)
        results_buffer.append(f"{n},{','.join((str(x) for x in crop_coords))},FALSE")

        # If some mask has been cropped out, find the locations of it
        cropped = transform.crop(mask, crop_coords, crop_size, centred=False)
        if cropped.sum() != mask.sum():
            bounds = [
                transform.start_and_end(a, b, start_from_loc=c)
                for (a, b, c) in zip(crop_coords, crop_size, [True, False, False])
            ]
            # Probably just find the co-ordinates for the crop, then
            # find which white pixels lie outside this
            oob_pixels = _find_excluded_pixels(mask, bounds)
            warning_buffer.append(
                f"""Some mask has been cropped out for {path}:
                    {oob_pixels.shape[0]} pixels cropped out:
                    {' '.join(repr(list(x)) for x in oob_pixels)}
                    From bounds {bounds}
                 """
            )

    # Print the results
    print("\n".join(results_buffer))

    # Emit any warnings
    for message in warning_buffer:
        warnings.warn(message)


if __name__ == "__main__":
    main()
