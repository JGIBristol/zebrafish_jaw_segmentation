"""
Training data for the jaw localisation model.

"""

import pathlib


def downsampled_dicom_path(dicom_path: pathlib.Path) -> pathlib.Path:
    """
    Get the path to the downsampled dicom file for a given dicom path.
    """
    parts = dicom_path.parts

    i = parts.index("dicoms")
    downsampled_parts = parts[: i + 1] + ("downsampled_dicoms",) + parts[i + 1 :]
    return pathlib.Path(*downsampled_parts)
