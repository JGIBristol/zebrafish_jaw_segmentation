"""
Find how much of the DICOMs are occupied by jaw voxels, in both the cropped and
uncropped cases.

"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.images import io, transform
from fishjaw.util import util, files, metadata


def main():
    """
    Read in each DICOM, take the sum of the mask, and print some stats

    """
    config = util.userconf()
    window_size = transform.window_size(config)

    # Don't need to pass a config in if we're reading all the DICOMs
    paths = files.dicom_paths(None, "all")

    # We only want the complete jaws
    paths = [path for path in paths if "Training set 3 (base of jaw)" not in path.stem]

    means = []
    cropped_means = []
    ages = []
    for path in tqdm(paths):
        _, mask = io.read_dicom(path)

        n = int(path.stem.split("_", maxsplit=1)[-1])
        crop_coords = transform.centre(n)
        around_centre = transform.around_centre(n)

        ages.append(metadata.age(n))

        means.append(1000 * mask.mean())

        mask = transform.crop(mask, crop_coords, window_size, around_centre)
        cropped_means.append(1000 * mask.mean())

    # Per mil character
    print(f"Uncropped: {np.mean(means):.2f} +/- {np.std(means):.2f}\u2030")
    print(
        f"Cropped: {np.mean(cropped_means):.2f} +/- {np.std(cropped_means):.2f}\u2030"
    )

    plt.hist(cropped_means)
    plt.xlabel("Occupancy (\u2030)")
    plt.ylabel("Frequency")
    plt.title("Histogram jaw occupancy after cropping")
    plt.savefig(f"{files.script_out_dir()}/jaw_occupancy.png")


if __name__ == "__main__":
    main()
