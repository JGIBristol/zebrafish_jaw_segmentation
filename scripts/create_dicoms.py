"""
Create DICOM files from the data on RDSF so we have everything nicely paired up and in one place

"""

import pathlib
import datetime

import pydicom
import tifffile
import numpy as np
from tqdm import tqdm

from fishjaw.util import files


class Dicom:
    """
    Create a DICOM file from an image and label

    """

    def __init__(
        self, image_path: np.ndarray, label_path: np.ndarray, *, binarise: bool
    ):
        self.label = tifffile.imread(label_path)
        if binarise:
            # These are the labels that Wahab used
            self.label[np.isin(self.label, [2, 3, 4, 5])] = 1

        self.image = tifffile.imread(image_path)

        if self.image.shape != self.label.shape:
            raise ValueError(
                f"Image and label shape do not match: {self.image.shape} vs {self.label.shape}"
            )

        if not set(np.unique(self.label)) == {0, 1}:
            raise ValueError(f"Label must be binary, got {np.unique(self.label)}")

        self.fish_label = image_path.name.split(".")[0]

    def write(self, out_path: pathlib.Path):
        """
        Write to file

        """
        file_meta = pydicom.dataset.FileMetaDataset()
        ds = pydicom.dataset.FileDataset(
            str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128
        )

        # DICOM metadata
        ds.PatientName = self.fish_label
        ds.PatientID = self.fish_label
        ds.Modality = "CT"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SOPClassUID = pydicom.uid.CTImageStorage

        # Image data
        ds.NumberOfFrames, ds.Rows, ds.Columns = self.image.shape
        ds.PixelData = self.image.tobytes()

        # Ensure the pixel data type is set to 16-bit
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0 if self.image.dtype.kind == "u" else 1

        # Set required attributes for pixel data conversion
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # Set Window Center and Window Width, so that the image is displayed correctly
        min_pixel_value = np.min(self.image)
        max_pixel_value = np.max(self.image)
        window_center = (max_pixel_value + min_pixel_value) / 2
        window_width = max_pixel_value - min_pixel_value
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width

        # Label data
        private_creator_tag = 0x00BBB000
        ds.add_new(private_creator_tag, "LO", "LabelData")
        label_data_tag = 0x00BBB001
        ds.add_new(label_data_tag, "OB", self.label.tobytes())

        # More crap
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.ContentDate = str(datetime.date.today()).replace("-", "")
        ds.ContentTime = (
            str(datetime.datetime.now().time())
            .replace(":", "")
            .split(".", maxsplit=1)[0]
        )

        ds.save_as(out_path, write_like_original=False)
        print(f"Saved to {out_path}")


def create_wahab_dicoms() -> None:
    """
    Create DICOMs from Wahab's segmented images

    """
    label_paths = sorted(
        list(files.wahab_labels_dir().glob("*.tif")), key=lambda x: x.name
    )

    # Find the corresponding images
    wahab_tif_dir = files.wahab_3d_tifs_dir()
    for label_path in tqdm(
        label_paths[2:], desc="Creating DICOMs from Wahab's segmentations"
    ):
        img_path = wahab_tif_dir / f"ak_{label_path.name}"
        if not img_path.exists():
            print(
                f"Could not find corresponding image at {img_path} for label at {label_path}"
            )
            continue

        out_path = files.dicom_dir() / img_path.name.replace(".tif", ".dcm")

        if out_path.exists():
            print(f"Skipping {out_path}, already exists")
            continue

        try:
            # These contain different labels for the different bones
            dicom = Dicom(img_path, label_path, binarise=True)
        except ValueError as e:
            print(f"Skipping {img_path} and {label_path}: {e}")
            continue

        dicom.write(out_path)


def create_felix_second_dicoms():
    """
    Create DICOMs from Felix's second set of segmented images

    """
    # Find the paths for the labels
    label_paths = list(files.felix_labels_2_dir().glob("*.tif"))

    # Find the corresponding images
    wahab_tif_dir = files.wahab_3d_tifs_dir()
    for label_path in tqdm(
        label_paths, desc="Creating DICOMs from Felix's first segmentations"
    ):
        img_path = wahab_tif_dir / label_path.name.replace(".labels", "")
        if not img_path.exists():
            raise FileNotFoundError(
                f"Could not find corresponding image for {label_path}"
            )

        out_path = files.dicom_dir() / img_path.name.replace(".tif", ".dcm")

        if out_path.exists():
            print(f"Skipping {out_path}, already exists")
            continue

        try:
            dicom = Dicom(img_path, label_path, binarise=False)
        except ValueError as e:
            print(f"Skipping {img_path} and {label_path}: {e}")
            continue

        dicom.write(out_path)


def main():
    """
    Get the images and labels, create DICOM files and save them to disk

    """
    if not files.dicom_dir().is_dir():
        files.dicom_dir().mkdir()

    # We should really check here that there's no overlap between Felix and Wahab's images

    create_wahab_dicoms()
    create_felix_second_dicoms()


if __name__ == "__main__":
    main()
