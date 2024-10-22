"""
Create DICOM files from the data on RDSF so we have everything nicely paired up and in one place

"""

import pathlib
import datetime
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
    binarise: bool

    def __post_init__(self):
        self.label = tifffile.imread(self.label_path)
        if self.binarise:
            # These are the labels that Wahab used
            self.label[np.isin(self.label, [2, 3, 4, 5])] = 1

        self.image = tifffile.imread(self.image_path)

        if self.image.shape != self.label.shape:
            raise ValueError(
                f"Image and label shape do not match: {self.image.shape} vs {self.label.shape}"
            )

        if not set(np.unique(self.label)) == {0, 1}:
            raise ValueError(f"Label must be binary, got {np.unique(self.label)}")

        self.fish_label = self.image_path.name.split(".")[0]


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


def create_set_1(config: dict) -> None:
    """
    Create DICOMs from Training set 1 - Wahab's segmented images

    """
    label_dir = config["rdsf_dir"] / pathlib.Path(util.config()["label_dirs"][0])
    label_paths = sorted(
        list(label_dir.glob("*.tif")),
        key=lambda x: x.name,
    )

    dicom_dir = files.dicom_dirs()[0]
    if not dicom_dir.is_dir():
        dicom_dir.mkdir(parents=True)

    # Find the corresponding CT scan images
    img_paths = [files.image_path(label_path) for label_path in label_paths]
    for label_path, img_path in tqdm(
        zip(label_paths, img_paths), desc="Creating DICOMs from Wahab's segmentations"
    ):
        if not img_path.exists():
            print(
                f"Image at {img_path} not found, but {label_path} was"
            )
            continue

        dicom_path = dicom_dir / img_path.name.replace(".tif", ".dcm")

        if dicom_path.exists():
            print(f"Skipping {dicom_path}, already exists")
            continue

        try:
            # These contain different labels for the different bones
            dicom = Dicom(img_path, label_path, binarise=True)
        except ValueError as e:
            print(f"Skipping {img_path} and {label_path}: {e}")
            continue

        write_dicom(dicom, dicom_path)


def create_felix_second_dicoms(config: dict):
    """
    Create DICOMs from Felix's second set of segmented images

    """
    # Find the paths for the labels
    label_paths = list(files.felix_labels_2_dir(config).glob("*.tif"))

    # Find the corresponding images
    wahab_tif_dir = files.wahab_3d_tifs_dir(config)
    for label_path in tqdm(
        label_paths, desc="Creating DICOMs from Felix's first segmentations"
    ):
        img_path = wahab_tif_dir / label_path.name.replace(".labels", "")
        if not img_path.exists():
            raise FileNotFoundError(
                f"Could not find corresponding image for {label_path}"
            )

        out_path = files.dicom_dir(config) / img_path.name.replace(".tif", ".dcm")

        if out_path.exists():
            print(f"Skipping {out_path}, already exists")
            continue

        try:
            dicom = Dicom(img_path, label_path, binarise=False)
        except ValueError as e:
            print(f"Skipping {img_path} and {label_path}: {e}")
            continue

        write_dicom(dicom, out_path)


def main():
    """
    Get the images and labels, create DICOM files and save them to disk

    """
    config = util.userconf()

    # We should really check here that there's no overlap between Felix and Wahab's images

    create_set_1(config)
    create_felix_second_dicoms(config)


if __name__ == "__main__":
    main()
