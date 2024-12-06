"""
I'm going insane so im tracking this in git

The tiffs that I've converted from 2d->3d myself are misaligned with wahab's masks.
Lets just try flipping the dicom in every possible way to see if we can get it to align

"""

from itertools import permutations

import tqdm
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from fishjaw.visualisation import images_3d
from fishjaw.util import files
from fishjaw.images import transform


def rotate_3d(array):
    """
    Generate all 48 unique orientations of a 3D array.

    :param array: 3D NumPy array to rotate.
    :return: Generator yielding all 24 3D NumPy array orientations.
    """
    axis_permutations = list(permutations((0, 1, 2)))

    # Define all possible flips
    flip_combinations = [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]

    for axes in axis_permutations:
        for flip in flip_combinations:
            transformed = array.copy()
            if flip[0]:
                transformed = np.flip(transformed, axis=0)
            if flip[1]:
                transformed = np.flip(transformed, axis=1)
            if flip[2]:
                transformed = np.flip(transformed, axis=2)
            transformed = np.transpose(transformed, axes=axes)
            for k in range(4):  # 4 rotations (0, 90, 180, 270 degrees)
                transformed = np.rot90(transformed, k=k, axes=(axes[1], axes[2]))

                for transpose in (True, False):
                    if transpose:
                        transformed = np.transpose(transformed, (0, 2, 1))
                    if transformed.shape == array.shape:
                        yield f"rot90_{k=}_{axes=}_{flip=}_{transpose=}", transformed


def main():
    """
    Read in the image and mask, do every possible flip/rotation and find the sum of their product
    Maybe the maximum one will be the correct one

    """
    ct = tifffile.imread(
        "/home/mh19137/zebrafish_rdsf/1Felix and Rich make models/wahabs_scans/351.tif"
    )
    mask = tifffile.imread(
        "/home/mh19137/zebrafish_rdsf/1Felix and Rich make models/Training dataset Tiffs/Training set 1/ak_351.labels.tif"
    )

    out_dir = files.boring_script_out_dir() / "align_tiffs"
    if not out_dir.is_dir():
        out_dir.mkdir()

    # We will be cropping
    crop_centre = 1667, 326, 276
    crop_size = 200, 200, 200
    cropped_mask = transform.crop(mask, crop_centre, crop_size, centred=True)

    # Build up a dict of the overlap
    results = {}

    for transform_str, array in tqdm.tqdm(rotate_3d(ct), total=96):
        results[transform_str] = np.sum(array * mask > 0)

        # Plot the array and mask
        cropped = transform.crop(array, crop_centre, crop_size, centred=True)

        fig, _ = images_3d.plot_slices(cropped, cropped_mask)
        fig.savefig(out_dir / f"{transform_str.replace(' ', '_').replace(',','_')}.png")
        plt.close(fig)

    # Sort the results and print
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    for key in results:
        print(f"{key}: {results[key]}")


if __name__ == "__main__":
    main()
