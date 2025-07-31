"""
Train a model to locate the zebrafish jaw.

We'll want to do this because we want to segment the jaw out from a CT scan of the
whole fish - the first step will be to crop the jaw out from it, and then we can
use the model trained in `train_model.py` to segment the jaw.

"""

import pathlib
import argparse

from fishjaw.localisation import data


def main():
    """
    Read (cached) downsampled dicoms, init a model and train it to localise the jaw.

    The jaw centre will be the centroid of the segmentation mask; we will use a heatmap
    with a gradually shrinking kernel to train the model. Then we will recover
    the jaw centre from the heatmap by convolving to find its centre.

    """
    dicom_paths = sorted(
        list((pathlib.Path("dicoms") / "Training set 2").glob("*.dcm"))
        + list(
            (
                pathlib.Path("dicoms") / "Training set 4 (Wahab resegmented by felix)"
            ).glob("*.dcm")
        )
    )

    downsampled_paths = [data.downsampled_dicom_path(p) for p in dicom_paths]


if __name__ == "__main__":
    main()
