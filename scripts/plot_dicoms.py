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


def plot_dicom(
    dicom_path: pathlib.Path, window_size: tuple[int, int, int] | None = None
):
    """
    Given a path to a DICOM file, plot it and save

    """
    out_path = dicom_path.with_suffix(".png")

    image, label = io.read_dicom(dicom_path)

    # Optionally crop
    if window_size is not None:
        n = files.dicompath_n(dicom_path)

        centre = transform.centre(n)
        around_centre = transform.around_centre(n)

        image = transform.crop(image, centre, window_size, around_centre)
        label = transform.crop(label, centre, window_size, around_centre)

    # Plot the slices
    fig, _ = images_3d.plot_slices(image, mask=label)

    fig.suptitle(str(dicom_path))
    fig.tight_layout()

    fig.savefig(out_path)
    plt.close(fig)


def main(*, crop: bool) -> None:
    """
    Plot the DICOMs that we've cached

    """
    config = util.userconf()
    for dicom_path in tqdm(files.dicom_paths(config, "all")):
        plot_dicom(dicom_path, transform.window_size(config) if crop else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot DICOM files that have been created with create_dicoms.py"
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop the images to the bounding box of the label."
        "Uses the window size in userconf.yml",
    )
    main(**vars(parser.parse_args()))
