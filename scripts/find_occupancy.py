"""
Find how much of the DICOMs are occupied by jaw voxels, in both the cropped and
uncropped cases.

"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.images import io, transform
from fishjaw.util import util, files, metadata


def _hist(cropped_means: list[float]) -> None:
    """
    Make a histogram of the cropped means

    """
    fig, axis = plt.subplots()

    axis.hist(cropped_means)
    axis.set_xlabel("Occupancy (\u2030)")
    axis.set_ylabel("Frequency")
    axis.set_title("Histogram jaw occupancy after cropping")

    fig.savefig(f"{files.script_out_dir()}/jaw_occupancy.png")
    plt.close(fig)


def _scatter(cropped_means, ages) -> None:
    """
    Make plots of mean size against age

    """
    fig, axis = plt.subplots()

    axis.plot(ages, cropped_means, "o")

    pts = axis.get_xlim()

    def line(x, a, b):
        """Straight line"""
        return a * x + b

    popt = np.polyfit(ages, cropped_means, 1)
    axis.plot(pts, [line(pt, *popt) for pt in pts], "r")
    axis.set_xlim(pts)

    axis.set_title("Cropped Jaw Occupancy vs Age")

    axis.set_xlabel("Age (months)")
    axis.set_ylabel("Occupancy (\u2030)")

    fig.tight_layout()
    fig.savefig(f"{files.script_out_dir()}/jaw_occupancy_age.png")


def main():
    """
    Read in each DICOM, take the sum of the mask, and print some stats

    """
    config = util.userconf()
    window_size = transform.window_size(config)

    # Don't need to pass a config in if we're reading all the DICOMs
    paths = files.dicom_paths(None, "all")

    # We only want the complete jaws
    paths = [path for path in paths if "Training set 3 (base of jaw)" not in str(path)]

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

    _hist(cropped_means)
    _scatter(cropped_means, ages)


if __name__ == "__main__":
    main()
