"""
Create DICOM files from the data on RDSF so we have everything nicely paired up and in one place

NB this script requires some things to be hard coded into the config file, namely:
dicom_dirs:
  - "/home/mh19137/zebrafish_jaw_segmentation/dicoms/Training set 1/"
  - "/home/mh19137/zebrafish_jaw_segmentation/dicoms/Training set 2/"
  - "/home/mh19137/zebrafish_jaw_segmentation/dicoms/Training set 3 (base of jaw)/"
TODO fix this

This script works by reading in the TIFF labels and images that live on the RDSF and creating
DICOM files from them, which it then saves to disk. This is useful because it keeps the image
and label together in the same file (we don't want to get confused between files; they may have
different numbering schemes, may be partially labelled, and/or may live in different places on
the RDSF).
It is also much faster to read in a DICOM file than it is to read the TIFFs from the RDSF (which
is a network drive) every time we want to perform inference, train a model, plot something, etc.
There are three basic categories of files at the moment:
 - Training set 1 - Wahab's segmented images
 - Training set 2 - the first set of images that Felix segmented
 - Training set 3 - the second set of images that Felix segmented; only the rear part of the jaw
                    has been segmented.

This script just translates Training sets 2 and 3 directly to DICOMs.
For Training set 1, Wahab has segmented both the jawbones we're interested in and the quadrate
bones - these have been given different labels, so we ignore the quadrates and keep only the
other ones.

"""

import pathlib
import datetime
import argparse
from typing import Any
from dataclasses import dataclass

import pydicom
import tifffile
import numpy as np
from tqdm import tqdm

from fishjaw.util import files, util
from fishjaw.model import data


def _get_n(label_path: pathlib.Path) -> int:
    """
    Get the fish number from the label path

    """
    parent = label_path.parent.name
    stem = label_path.stem
    if "Wahab resegmented by felix" in parent:
        # For training set 4 (Wahab's one), read it from the filename and convert
        # to the new_n scheme
        old_n = int(stem[4:].split("_")[0])
        return files.oldn2newn()[old_n]

    # The first number in the filename for training sets 2 and 3
    return int(stem.split(".")[0][3:])


def create_dicoms(
    config: dict[str, Any],
    dir_index: int,
    dry_run: bool,
    *,
    ignore: set[int] | None = None,
) -> None:
    """
    Create DICOMs from images and segmentation masks

    """
    if ignore is None:
        ignore = set()

    # Get paths to the labels
    label_dir = config["rdsf_dir"] / pathlib.Path(
        util.config()["label_dirs"][dir_index]
    )
    label_paths = sorted(
        list(label_dir.glob("*.tif")),
        key=lambda x: x.name,
    )
    if len(label_paths) == 0:
        raise ValueError(f"No images found in {label_dir}")

    # Get paths to the images
    # This will be a list of directories for the 2D images and a list of files for the 3D images
    img_paths = (
        [files.get_2d_tif_dir(config, label_path) for label_path in label_paths]
        if dir_index == 2
        else [files.get_3d_tif(label_path) for label_path in label_paths]
    )

    # Create the directory to store the DICOMs
    dicom_dir = files.dicom_dirs(config)[dir_index]
    if not dicom_dir.is_dir():
        dicom_dir.mkdir(parents=True)

    for label_path, img_path in tqdm(
        zip(label_paths, img_paths), total=len(label_paths)
    ):
        if not img_path.exists():
            raise RuntimeError(f"Image at {img_path} not found, but {label_path} was")

        n = _get_n(label_path)
        if n in ignore:
            print(f"Skipping {label_path}, fish number in ignore set")
            continue

        dicom_path = dicom_dir / f"{n}.dcm"

        if dicom_path.exists():
            print(f"Skipping {dicom_path}, already exists")
            continue

        if dry_run:
            print(f"Would write {dicom_path}")
        else:
            try:
                # These contain different labels for the different bones
                dicom = data.Dicom(img_path, label_path)
            except ValueError as e:
                print(f"Skipping {img_path} and {label_path}: {e}")
                continue

            data.write_dicom(dicom, dicom_path)


def main(dry_run: bool):
    """
    Get the images and labels, create DICOM files and save them to disk

    This is a pretty ugly function that passes an index around, instead of anything more
    intuitive, but its ok as long as you can match up the index here (0, 1, 2) with the
    name of the directories that we're reading from (in config.yml) and the ones we're
    writing to (in userconf.yml).

    """
    config = util.userconf()

    # Training set 2 - Felix's segmented images
    create_dicoms(config, 0, dry_run, ignore=files.broken_dicoms())

    # Training set 3 - Felix's segmented rear jaw only
    # Some might be duplicated between the different sets
    # So exclude the duplicates here
    # Also, some of the shapes don't match up with the labels, so exclude those too
    create_dicoms(
        config, 1, dry_run, ignore=files.duplicate_dicoms() | files.broken_dicoms()
    )

    # Training set 4
    # Felix's resegmentations from Wahab's images - should be no duplicates
    create_dicoms(config, 2, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DICOM files from TIFFs")

    parser.add_argument("--dry-run", action="store_true", help="Don't write any files")
    main(**vars(parser.parse_args()))
