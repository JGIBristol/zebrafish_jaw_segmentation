"""
Training data for the jaw localisation model.

"""

import pathlib
import datetime

import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from scipy.ndimage import zoom, center_of_mass, gaussian_filter


class HeatmapDataset(Dataset):
    """
    Initialise a dataset for training the model - images and heatmaps.

    Calculates the heatmaps by putting a Gaussian at the centroid of each mask.
    """

    def __init__(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        sigma: float,
    ):
        self.img_shape = images[0].shape

        assert all(
            img.shape == mask.shape for img, mask in zip(images, masks)
        ), "All images and masks must have the same shape"
        assert len(images) == len(masks), "Number of images and masks must match"
        assert all(
            img.shape == self.img_shape for img in images
        ), "All images must have the same shape"

        self.data = torch.tensor(
            np.array(images, dtype=np.float32), dtype=torch.float32
        ).unsqueeze(1)

        # Find the approx centroids of the masks
        # (we'll use these to create heatmaps later)
        self._centroids = [tuple(map(int, center_of_mass(mask))) for mask in masks]

        self.set_heatmaps(sigma)

    def _create_heatmap(
        self, centroid: tuple[int, int, int], sigma: float
    ) -> np.ndarray:
        """
        Create a heatmap from a centroid by placing a Gaussian at the centroid.
        """
        heatmap = np.zeros(self.img_shape, dtype=np.float32)
        heatmap[centroid] = 1.0
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        assert np.allclose(
            np.sum(heatmap), 1.0
        ), f"Heatmap should sum to 1; is the centroid to close to the edge? {centroid=}, {sigma=}, {self.img_shape=}"

        return heatmap

    def set_heatmaps(self, sigma: float) -> None:
        """
        Reset the sigma for the heatmaps and recalculate them.
        """
        heatmaps = [
            self._create_heatmap(centroid, sigma) for centroid in self._centroids
        ]

        self.heatmaps = torch.tensor(np.array(heatmaps), dtype=torch.float32).unsqueeze(
            1
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Doesn't send the data to a device"""
        return self.data[idx], self.heatmaps[idx]


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


def scale_prediction_up(
    predicted_coords: tuple[int, int, int],
    sf: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Scale the prediction back up"""
    return tuple(int(coord / _sf) for coord, _sf in zip(predicted_coords, sf))
