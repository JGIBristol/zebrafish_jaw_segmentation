"""
Reading + writing images

"""

import pathlib

import pydicom
import numpy as np

from ..util import util


def patch_size() -> tuple[int, int, int]:
    """
    Read the patch size from the config file

    :return: The patch size ZYX

    """

    return tuple(int(x) for x in util.userconf()["patch_size"].split(","))


def read_dicom(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read an image and label from a DICOM file

    :param path: Path to the DICOM file

    :returns: The image
    :returns: The label

    """
    # Read the image
    dataset = pydicom.dcmread(path)
    image = dataset.pixel_array

    # Read the label
    # It's probably bad that these are hard-coded and not registered anywhere
    private_creator_tag = 0x00BBB000
    label_data_tag = 0x00BBB001
    if private_creator_tag not in dataset and label_data_tag not in dataset:
        raise AttributeError(f"No label data found for {path}")

    label = np.frombuffer(dataset[label_data_tag].value, dtype=np.uint8).reshape(
        image.shape
    )

    return image, label
