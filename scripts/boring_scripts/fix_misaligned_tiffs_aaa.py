"""
I'm going insane so im tracking this in git

The tiffs that I've converted from 2d->3d myself are misaligned with wahab's masks.
Lets just try flipping the dicom in every possible way to see if we can get it to align

"""

from itertools import permutations

import tqdm
import tifffile
import numpy as np


def rotate_3d(array):
    """
    Generate all 48 unique orientations of a 3D array.

    :param array: 3D NumPy array to rotate.
    :return: Generator yielding all 24 3D NumPy array orientations.
    """
    pbar = tqdm.tqdm(total=48)

    # Start with the identity (no rotation)
    yield "identity", array
    pbar.update(1)

    # Define all possible axis permutations
    axis_permutations = list(permutations((0, 1, 2), 2))

    # For each axis pair, perform 90-degree rotations (4 times)
    for axes in axis_permutations:
        for k in range(1, 4):  # k=1 to 3 rotates 90, 180, 270 degrees
            rotated = np.rot90(array, k=k, axes=axes)
            if rotated.shape != array.shape:
                continue

            yield f"rot90, {k=}, {axes=}", rotated
            pbar.update(1)

    # Additional rotations for full 3D coverage
    # Flip along each axis and apply the rotations again
    for flip_axis in range(3):  # 0, 1, 2 correspond to x, y, z axes
        flipped_array = np.flip(array, axis=flip_axis)
        if flipped_array.shape != array.shape:
            continue
        yield f"flipped, {flip_axis=}", flipped_array  # Include the flipped version itself
        pbar.update(1)

        for axes in axis_permutations:
            for k in range(1, 4):
                rotated = np.rot90(flipped_array, k=k, axes=axes)
                if rotated.shape != array.shape:
                    continue
                yield f"flipped {flip_axis=}, rot90, {k=}, {axes=}", rotated
                pbar.update(1)


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

    results = {}
    for transform, array in rotate_3d(ct):
        results[transform] = np.sum(array * mask)

    results = {
        k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)
    }
    for key in results:
        print(f"{key}: {results[key]}")


if __name__ == "__main__":
    main()
