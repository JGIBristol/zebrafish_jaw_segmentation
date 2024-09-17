"""
Plot DICOM files that were created by create_dicoms.py

"""

import pathlib
import argparse

from tqdm import tqdm
from matplotlib import pyplot as plt

from fishjaw.util import files, util
from fishjaw.visualisation import images_3d
from fishjaw.images import transform, io


def plot_dicom(dicom_path: pathlib.Path, crop: bool, window_size: tuple[int, int, int]):
    """
    Given a path to a DICOM file, plot it and save

    """
    out_path = dicom_path.with_suffix(".png")

    image, label = io.read_dicom(dicom_path)

    # Optionally crop
    if crop:
        # Extract N from the filename
        n = int(dicom_path.stem.split("_", maxsplit=1)[-1])

        centre = transform.centre(n)

        image = transform.crop(image, centre, window_size)
        label = transform.crop(label, centre, window_size)

    # Plot the slices
    fig, _ = images_3d.plot_slices(image, mask=label)

    fig.suptitle(str(dicom_path))
    fig.tight_layout()

    fig.savefig(out_path)
    plt.close(fig)


def main(*, crop: bool):
    """
    Plot the DICOMs that we've cached

    """
    config = util.userconf()
    dicom_dir = files.dicom_dir(config)

    window_size = (160, 160, 160)
    for dicom_path in tqdm(sorted(list(dicom_dir.glob("*.dcm")))):
        plot_dicom(dicom_path, crop, window_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot DICOM files that have been created with create_dicoms.py"
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop the images to the bounding box of the label",
    )
    main(**vars(parser.parse_args()))
