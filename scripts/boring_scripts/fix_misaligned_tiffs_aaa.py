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


def rotations24(polycube, flipped=False):
    """List all 24 rotations of the given 3d array"""

    prefix = "flipped" if flipped else ""

    def rotations4(polycube, axes):
        """List the four rotations of the given 3d array in the plane spanned by the given axes."""
        for i in range(4):
            yield f"{prefix}{axes=}_{i=}", np.rot90(polycube, i, axes)

    # 4 rotations about axis 0
    yield from rotations4(polycube, (1, 2))

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    yield from rotations4(np.rot90(polycube, 2, axes=(0, 2)), (1, 2))

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    yield from rotations4(np.rot90(polycube, axes=(0, 2)), (0, 1))
    yield from rotations4(np.rot90(polycube, -1, axes=(0, 2)), (0, 1))

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    yield from rotations4(np.rot90(polycube, axes=(0, 1)), (0, 2))
    yield from rotations4(np.rot90(polycube, -1, axes=(0, 1)), (0, 2))

def rotations48(polycube):
    yield from rotations24(polycube)
    flipped = np.flip(polycube, axis=0)
    yield from rotations24(flipped, flipped=True)


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

    for transform_str, array in tqdm.tqdm(rotations48(ct), total=48):
        if array.shape != ct.shape:
            continue
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
