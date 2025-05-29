"""
Data for transfer learning
"""

import torchio as tio


def _quadrate_paths() -> dict[str, str]:
    """
    Return a mapping from the quadrate data paths to their respective labels
    on the RDSF.
    """


def _cache_quadrates() -> None:
    """
    Save the quadrate data to DICOM
    """


def quadrate_data() -> tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.Subject]:
    """
    Get the train/validation/test quadrate data

    Cropped around the centre of each label
    """
