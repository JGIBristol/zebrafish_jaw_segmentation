"""
Crop and plot a DICOM created by create_dicoms.py.

This is intended to be a visual check that we've cropped the right region out -
this isn't that important for the complete jaws which we crop a bounding box around, but
it is important for the partial jaws where the full DICOM contains some unlabelled jaw.
We don't want any of this unlabelled jaw to be included in the cropped image, so we want to check
that the last cropped slice contains labelled jaw and the one beyond it does not.

"""

import pathlib
import argparse

from tqdm import tqdm
from matplotlib import pyplot as plt

from fishjaw.util import files, util
from fishjaw.visualisation import images_3d
from fishjaw.images import transform, io


def _get_dicom(n: int) -> pathlib.Path:
    """
    Get the path to the dicom given the number

    """


def _plot_slices(
    dicom_path: pathlib.Path,
    window_size: tuple[int, int, int],
    crop_coords: tuple[int, int, int],
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Given a path to a DICOM file, plot it and save

    """

    # image, label = io.read_dicom(dicom_path)

    # # Optionally crop
    # if window_size is not None:
    #     # Extract N from the filename
    #     n = int(dicom_path.stem.split("_", maxsplit=1)[-1])

    #     centre = transform.centre(n)
    #     around_centre = transform.around_centre(n)

    #     image = transform.crop(image, centre, window_size, around_centre)
    #     label = transform.crop(label, centre, window_size, around_centre)

    # # Plot the slices
    # fig, _ = images_3d.plot_slices(image, mask=label)

    # fig.suptitle(str(dicom_path))
    # fig.tight_layout()

    # fig.savefig(out_path)
    # plt.close(fig)


def main(*, n: bool):
    """
    Plot the DICOMs that we've cached

    """

    # Get the right DICOM
    dicom_path = _get_dicom(n)

    # Get the window size and crop co-ordinates
    config = util.userconf()
    window_size = transform.window_size(config)
    crop_coords = transform.centre(n)

    # If we're cropping around the centre, just show the slices as they are
    # If we're cropping from the edge, increment the z co-ordinate by 1 and the z window size by 1
    # So that we can also see the slice beyond the cropped region
    crop_from_edge = not transform.around_centre(n)
    if crop_from_edge:
        window_size = (window_size[0] + 1, *window_size[1:])

    fig, axes = _plot_slices(dicom_path, window_size, crop_coords)

    if crop_from_edge:
        fig.suptitle("Plotted from edge")
        axes[0].set_title("Slice before ROI")
        axes[-1].set_title("Slice after ROI")
    else:
        fig.suptitle("Plotted around centre")

    out_path = (
        pathlib.Path(util.config()["script_output"]) / dicom_path.stem / "_cropped.png"
    )
    fig.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Show the result of the cropping (defined in data/jaw_centres.csv) on the DICOMs.
        Intended to be a visual check that we've cropped the right region out.
        """
    )
    parser.add_argument(
        "n",
        help="Image number. Check the dicoms/ directory for the available numbers.",
        type=int,
    )
    main(**vars(parser.parse_args()))
