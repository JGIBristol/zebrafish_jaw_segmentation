"""
Training data for the jaw localisation model.

"""

import pathlib
import datetime

import pydicom
import numpy as np
from scipy.ndimage import zoom


def downsampled_dicom_path(dicom_path: pathlib.Path) -> pathlib.Path:
    """
    Get the path to the downsampled dicom file for a given dicom path.
    """
    parts = dicom_path.parts

    i = parts.index("dicoms")
    downsampled_parts = parts[: i + 1] + ("downsampled_dicoms",) + parts[i + 1 :]
    return pathlib.Path(*downsampled_parts)


def scale_factor(
    orig_shape: tuple[int, int, int], target_shape: tuple[int, int, int]
) -> tuple[float, float, float]:
    """
    Get the scale factor for downsampling
    """
    return tuple(
        target_dim / orig_dim for target_dim, orig_dim in zip(target_shape, orig_shape)
    )


def downsample(
    image: np.ndarray, mask: np.ndarray, target_shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample the image and mask to the provided target shape

    Returns resized img/mask

    """
    assert image.shape == mask.shape, "Image and mask must have the same shape"

    sf = scale_factor(image.shape, target_shape)

    # Cubic interpolation for images, nearest neighbour for masks
    resized_image = zoom(image, sf, order=3)
    resized_mask = zoom(mask, sf, order=0)

    return resized_image, resized_mask


def write_dicom(image: np.ndarray, mask: np.ndarray, out_path: pathlib.Path) -> None:
    """
    Write a dicom to file

    Bad duplicate code but im tired

    :param dicom: Dicom object
    :param out_path: Path to save the dicom to

    """
    file_meta = pydicom.dataset.FileMetaDataset()
    ds = pydicom.dataset.FileDataset(
        str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128
    )

    # DICOM metadata
    ds.PatientName = None
    ds.PatientID = None
    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = pydicom.uid.CTImageStorage

    # Image data
    ds.NumberOfFrames, ds.Rows, ds.Columns = image.shape
    ds.PixelData = image.tobytes()

    # Ensure the pixel data type is set to 16-bit
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0 if image.dtype.kind == "u" else 1

    # Set required attributes for pixel data conversion
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Set Window Center and Window Width, so that the image is displayed correctly
    min_pixel_value = np.min(image)
    max_pixel_value = np.max(image)
    window_center = (max_pixel_value + min_pixel_value) / 2
    window_width = max_pixel_value - min_pixel_value
    ds.WindowCenter = window_center
    ds.WindowWidth = window_width

    # Label data
    private_creator_tag = 0x00BBB000
    ds.add_new(private_creator_tag, "LO", "LabelData")
    label_data_tag = 0x00BBB001
    ds.add_new(label_data_tag, "OB", mask.tobytes())

    # More crap
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.ContentDate = str(datetime.date.today()).replace("-", "")
    ds.ContentTime = (
        str(datetime.datetime.now().time()).replace(":", "").split(".", maxsplit=1)[0]
    )

    ds.save_as(out_path, write_like_original=False)
