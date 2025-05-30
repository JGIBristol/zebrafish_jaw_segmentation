"""
Data for transfer learning
"""

import re
import pathlib

import torch
import torchio as tio
from tqdm import tqdm
from scipy.ndimage import center_of_mass

from fishjaw.util import files
from fishjaw.images import io, transform
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


def _dicom_path(config: dict, label_path: pathlib.Path) -> pathlib.Path:
    """
    Get the path to the DICOM file for a given label path
    """
    return _quadrate_dir(config) / f"{label_path.stem}.dcm"


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

    dicom_path = _dicom_path(config, label_path)

    if dicom_path.exists():
        raise FileExistsError(
            f"Quadrate DICOM {dicom_path} already exists, not overwriting"
        )

    data.write_dicom(data.Dicom(img_path, label_path, binarise=True), dicom_path)


def quadrate_data(
    config: dict,
) -> tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.Subject]:
    """
    Get the train/validation/test quadrate data

    Cropped around the centre of each label, using the crop size from the config.
    Caches the data as DICOMs for faster repeated access - the first time you run this,
    it will be much slower than later runs.
    """
    # Get a mapping from label paths to image paths
    paths = _quadrate_paths(config)

    # Build up a list of subjects
    subjects = []
    for label_path, img_path in tqdm(
        paths.items(), total=len(paths), desc="Loading quadrate data"
    ):
        dicom_path = _dicom_path(config, label_path)
        if not dicom_path.exists():
            _cache_quadrate(config, img_path, label_path)

        # Read the image and label
        image, mask = io.read_dicom(dicom_path)

        # Get the crop centre from the centre of mass of the label
        centroid = tuple(round(x) for x in center_of_mass(mask))

        # Create the subject
        crop_size = transform.window_size(config)
        image = transform.crop(image, centroid, crop_size, centred=True)
        image = data.ints2float(image.copy())

        mask = transform.crop(mask, centroid, crop_size, centred=True)
        mask = mask.copy()

        subjects.append(
            tio.Subject(
                image=tio.Image(
                    tensor=data._add_dimension(image, dtype=torch.float32),
                    type=tio.INTENSITY,
                ),
                label=tio.Image(
                    tensor=data._add_dimension(mask, dtype=torch.uint8), type=tio.LABEL
                ),
            )
        )

    # Create datasets
    train_subjects = tio.SubjectsDataset(
        subjects[:-2], transform=data._transforms(config["transforms"])
    )
    val_subjects = tio.SubjectsDataset(
        subjects[-2:-1], transform=data._transforms(config["transforms"])
    )
    test_subject = tio.Subject(subjects[-1])

    return train_subjects, val_subjects, test_subject
