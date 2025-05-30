"""
Data for transfer learning
"""

import re
import pathlib

import torchio as tio

from fishjaw.util import files
from fishjaw.model import data


def _quadrate_dir(config: dict) -> pathlib.Path:
    """
    Get the directory where the quadrate data is stored
    """
    return pathlib.Path(config["quadrate_dir"])


def _quadrate_paths(config: dict) -> dict[pathlib.Path, pathlib.Path]:
    """
    Return a mapping from the quadrate data paths to their respective labels
    on the RDSF.

    :param config: from userconf.yml
    """
    labels = (
        config["rdsf_dir"]
        / pathlib.Path(
            "1Felix and Rich make models/Training dataset Tiffs/Training set 1"
        )
    ).glob("*.tif")

    # Remove these ones, since the 3D tifs dont exist
    bad_labels = re.compile(r"(351|401|420|441)")
    labels = [label for label in labels if not bad_labels.search(label.name)]
    imgs = [files.get_3d_tif(label_path) for label_path in labels]

    return dict(zip(labels, imgs, strict=True))


def _cache_quadrate(
    config: dict, img_path: pathlib.Path, label_path: pathlib.Path
) -> None:
    """
    Save the quadrate data to DICOM

    :param config: from userconf.yml
    :param label_path, img_path: abspaths to files

    :raises FileExistsError: if the DICOM already exists

    """
    quadrate_dir = _quadrate_dir(config)
    if not quadrate_dir.is_dir():
        quadrate_dir.mkdir(parents=True)

    dicom_path = quadrate_dir / f"{label_path.stem}.dcm"

    if dicom_path.exists():
        raise FileExistsError(
            f"Quadrate DICOM {dicom_path} already exists, not overwriting"
        )

    data.write_dicom(data.Dicom(img_path, label_path), dicom_path)


def quadrate_data() -> tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.Subject]:
    """
    Get the train/validation/test quadrate data

    Cropped around the centre of each label
    """
