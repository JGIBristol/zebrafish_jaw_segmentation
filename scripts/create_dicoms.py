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


@dataclass
class Dicom:
    """
    Create a DICOM file from an image and label

    """

    image_path: pathlib.Path
    label_path: pathlib.Path

    def __post_init__(self):
        self.label = tifffile.imread(self.label_path)

        # If we've been passed a directory, stack the images inside it
        self.image = (
            tifffile.imread(self.image_path)
            if self.image_path.is_file()
            else self._stack_files()
        )

        if self.image.shape != self.label.shape:
            raise ValueError(
                f"Image and label shape do not match: {self.image.shape} vs {self.label.shape}"
            )

        if not set(np.unique(self.label)) == {0, 1}:
            raise ValueError(f"Label must be binary, got {np.unique(self.label)}")

        self.fish_label = self.image_path.name.split(".")[0]

    def _stack_files(self) -> np.typing.NDArray:
        """
        Given a directory holding image files, return a stacked tiff

        """
        imgs = [
            tifffile.imread(img)
            for img in tqdm(
                sorted(self.image_path.glob("*.tiff")),
                desc=f"Reading from {self.image_path}",
            )
        ]
        return np.stack(imgs, axis=0)


def write_dicom(dicom: Dicom, out_path: pathlib.Path) -> None:
    """
    Write a dicom to file

    :param dicom: Dicom object
    :param out_path: Path to save the dicom to

    """
    file_meta = pydicom.dataset.FileMetaDataset()
    ds = pydicom.dataset.FileDataset(
        str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128
    )

    # DICOM metadata
    ds.PatientName = dicom.fish_label
    ds.PatientID = dicom.fish_label
    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = pydicom.uid.CTImageStorage

    # Image data
    ds.NumberOfFrames, ds.Rows, ds.Columns = dicom.image.shape
    ds.PixelData = dicom.image.tobytes()

    # Ensure the pixel data type is set to 16-bit
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0 if dicom.image.dtype.kind == "u" else 1

    # Set required attributes for pixel data conversion
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Set Window Center and Window Width, so that the image is displayed correctly
    min_pixel_value = np.min(dicom.image)
    max_pixel_value = np.max(dicom.image)
    window_center = (max_pixel_value + min_pixel_value) / 2
    window_width = max_pixel_value - min_pixel_value
    ds.WindowCenter = window_center
    ds.WindowWidth = window_width

    # Label data
    private_creator_tag = 0x00BBB000
    ds.add_new(private_creator_tag, "LO", "LabelData")
    label_data_tag = 0x00BBB001
    ds.add_new(label_data_tag, "OB", dicom.label.tobytes())

    # More crap
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.ContentDate = str(datetime.date.today()).replace("-", "")
    ds.ContentTime = (
        str(datetime.datetime.now().time()).replace(":", "").split(".", maxsplit=1)[0]
    )

    ds.save_as(out_path, write_like_original=False)


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
    img_paths = [files.image_path(label_path) for label_path in label_paths]

    # Create the directory to store the DICOMs
    dicom_dir = files.dicom_dirs(config)[dir_index]
    if not dicom_dir.is_dir():
        dicom_dir.mkdir(parents=True)

    for label_path, img_path in tqdm(
        zip(label_paths, img_paths), total=len(label_paths)
    ):
        if not img_path.exists():
            raise RuntimeError(f"Image at {img_path} not found, but {label_path} was")

        if ignore and int(label_path.stem.split(".")[0][3:]) in ignore:
            print(f"Skipping {label_path}, fish number in ignore set")
            continue

        dicom_path = dicom_dir / img_path.name.replace(".tif", ".dcm")

        if dicom_path.exists():
            print(f"Skipping {dicom_path}, already exists")
            continue

        if dry_run:
            print(f"Would write {dicom_path}")
        else:
            try:
                # These contain different labels for the different bones
                dicom = Dicom(img_path, label_path)
            except ValueError as e:
                print(f"Skipping {img_path} and {label_path}: {e}")
                continue

            write_dicom(dicom, dicom_path)


def main(dry_run: bool):
    """
    Get the images and labels, create DICOM files and save them to disk

    """
    config = util.userconf()

    # Training set 1 - Wahaab's segmented images
    create_dicoms(config, 0, dry_run, ignore=files.broken_dicoms())

    # Training set 2 - Felix's segmented images
    create_dicoms(config, 1, dry_run, ignore=files.broken_dicoms())

    # Some might be duplicated between the different sets; we only want
    # "Training set 3 - Felix's segmented rear jaw only" images
    # So exclude the duplicates
    create_dicoms(
        config, 2, dry_run, ignore=files.duplicate_dicoms() | files.broken_dicoms()
    )

    create_dicoms(config, 3, dry_run, ignore=files.broken_dicoms())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DICOM files from TIFFs")

    parser.add_argument("--dry-run", action="store_true", help="Don't write any files")
    main(**vars(parser.parse_args()))
