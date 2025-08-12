"""
Loading, pre-processing, etc. the data for the model

"""

import sys
import pathlib
import datetime
from typing import Any
from dataclasses import dataclass

import pydicom
import tifffile
import numpy as np
import torchio as tio
import torch
import torch.utils
from tqdm import tqdm

from ..images import io, transform
from ..util import files, util


@dataclass
class Dicom:
    """
    Create a DICOM file from an image and label

    """

    image_path: pathlib.Path
    label_path: pathlib.Path
    binarise: bool = False

    def __post_init__(self):
        """
        :param binarise: If True, binarise the label to only include elements where
                         the label == 4 or 5 (i.e. the quadrate in Wahab's labelling scheme)
        """
        self.label = tifffile.imread(self.label_path)
        if self.binarise:
            self.label = (self.label == 4) | (self.label == 5)

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


class DataConfig:
    """
    Create a DataConfig object, which holds the training, validation and test data

    This reads the data from the DICOMs stored on disk and creates DataLoaders for the
    training and validation data, and reserves a single Subject for testing.

    :param config: model training configuration, e.g. read from userconf.yml
    :param train_subjects: The training subjects
    :param val_subjects: The validation subjects

    """

    def __init__(
        self,
        config: dict,
        train_subjects: tio.SubjectsDataset,
        val_subjects: tio.SubjectsDataset,
    ):
        """
        Constructor

        Turns the training and validation subjects into DataLoaders, since they'll
        be passed to the model for training. Leaves the test subject alone, since
        it'll be passed to the model for testing.

        TODO the test subject should maybe also accept multiple, in case I want to test
        on multiple things. OR, it shouldn't be here at all - this is data that the model
        cares about, which doesn't include the testing subject

        """

        # Assign class variables
        self._train_data: tio.SubjectsLoader = self._train_val_loader(
            train_subjects, config, train=True
        )
        self._val_data: tio.SubjectsLoader = self._train_val_loader(
            val_subjects, config, train=False
        )

    def _train_val_loader(
        self,
        subjects: tio.SubjectsDataset,
        config: dict[str, Any],
        *,
        train: bool,
    ) -> tio.SubjectsLoader:
        """
        Create a dataloader from a SubjectsDataset

        Training data is shuffled and has the last batch dropped; validation data is not

        :param subjects: The dataset. Training data should have random transforms applied
        :param train: If we're training or not
        :param patch_size: The size of the patches to extract
        :param batch_size: The batch size

        :returns: The loader

        """
        # Get some info from the config
        patch_size = get_patch_size(config)
        batch_size = config["batch_size"]
        num_workers = config["num_workers"]

        shuffle = train is True
        drop_last = train is True

        patch_sampler = tio.UniformSampler(patch_size=patch_size)

        patches = tio.Queue(
            subjects,
            max_length=10000,  # Not sure if this matters
            samples_per_volume=1,
            sampler=patch_sampler,
            num_workers=num_workers,
            shuffle_patches=True,
            shuffle_subjects=True,
        )

        return tio.SubjectsLoader(
            patches,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=drop_last,
        )

    @property
    def train_data(self) -> tio.SubjectsLoader:
        """Get the training data"""
        return self._train_data

    @property
    def val_data(self) -> tio.SubjectsLoader:
        """Get the validation data"""
        return self._val_data


def ints2float(int_arr: np.ndarray) -> np.ndarray:
    """
    Scale an array from 16-bit integer values to float values in [0, 1]

    :param int_arr: The array to scale. Should be 16-bit integers - might be stored as a 32-bit
                    datatype, but the values should be in the range of a 16-bit integer

    :returns: The scaled array, with values between 0 and 1

    :raises: ValueError if the max value is less than the max value of a uint8 or greater
             than the max value of a uint16
    :raises: ValueError if the array is not of integer type

    """
    # Check that the values are in the right range
    if (max_val := int_arr.max()) < np.iinfo(np.uint8).max:
        raise ValueError(
            f"Max value {max_val} is less than can be stored by a uint8 - "
            "are you sure this is 16-bit data?"
        )
    if max_val > (uint16max := np.iinfo(np.uint16).max):
        raise ValueError(
            f"Max value {max_val} is greater than can be stored by a uint16 -"
            "are you sure this is 16-bit data?"
        )

    # Check that the values are of integer type
    if not np.issubdtype(int_arr.dtype, np.integer):
        raise ValueError(f"Array is not of integer type, but {int_arr.dtype}")

    return int_arr / uint16max


def _add_dimension(arr: np.ndarray, *, dtype: torch.dtype) -> np.ndarray:
    """
    Convert a numpy array to a torch tensor with an additional dimension

    :param arr: The array
    :param dtype: The data type to convert to. I'm not sure if we need this
    :returns: The tensor

    """
    return torch.as_tensor(arr, dtype=dtype).unsqueeze(0)


def cropped_dicom(
    dicom_path: pathlib.Path, window_size: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a DICOM file and crop it according to the jaw centres spreadsheet.

    Reads n from the filepath, finds the correct crop co-ordinates, reads the
    image and mask and crops them to the specified window size.

    :param dicom_path: Path to the DICOM file
    :param window_size: The size of the window to crop

    :returns: The cropped image and mask as numpy arrays
    :returns: The image and mask as numpy arrays
    :raises: CropOutOfBoundsError if the crop co-ordinates are out of bounds
             for the image or mask
    """
    # Load the image and mask from disk
    image, mask = io.read_dicom(dicom_path)

    # Find the co-ords and how to crop- either use this as the centre, or from the Z provided
    n = files.dicompath_n(dicom_path)
    crop_coords = transform.centre(n)
    around_centre = transform.around_centre(n)

    try:
        image = transform.crop(image, crop_coords, window_size, around_centre)
        mask = transform.crop(mask, crop_coords, window_size, around_centre)
    except transform.CropOutOfBoundsError as e:
        print(f"Error cropping {dicom_path}", file=sys.stderr)
        raise e

    return image, mask


def subject(dicom_path: pathlib.Path, window_size: tuple[int, int, int]) -> tio.Subject:
    """
    Create a subject from a DICOM file, cropping according to data/jaw_centres.csv

    :param dicom_path: Path to the DICOM file
    :param window_size: The size of the window to crop

    :returns: The subject

    """

    image, mask = cropped_dicom(dicom_path, window_size)

    # Convert to a float in [0, 1]
    # Need to copy since torch doesn't support non-writable tensors
    image = ints2float(image.copy())
    mask = mask.copy()

    return tio.Subject(
        image=tio.Image(
            tensor=_add_dimension(image, dtype=torch.float32), type=tio.INTENSITY
        ),
        label=tio.Image(tensor=_add_dimension(mask, dtype=torch.uint8), type=tio.LABEL),
    )


def imgs2subject(img: np.ndarray, label: np.ndarray) -> tio.Subject:
    """
    Create a subject from a greyscale image and a label
    """
    return tio.Subject(
        image=tio.Image(
            tensor=_add_dimension(ints2float(img.copy()), dtype=torch.float32),
            type=tio.INTENSITY,
        ),
        label=tio.Image(
            tensor=_add_dimension(label.copy(), dtype=torch.uint8), type=tio.LABEL
        ),
    )


def test_loader(
    patches: tio.GridSampler, *, batch_size: int
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for testing - this samples patches from the image in a grid
    such that we can reconstruct the entire image

    :param patches: patches from a subject, created with e.g. tio.GridSampler
    :param patch_size: The size of the patches to extract
    :param batch_size: The batch size

    :returns: The loader

    """
    return torch.utils.data.DataLoader(
        patches,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def load_transform(transform_name: str, args: dict) -> tio.transforms.Transform:
    """
    Load a transform from the configuration, which should be provided as a dict
    of {"name": {"arg1": value1, ...}}

    """
    return util.load_class(transform_name)(**args)


def _transforms(transform_dict: dict) -> tio.transforms.Transform:
    """
    Define the transforms to apply to the training data

    """
    return tio.Compose(
        [
            load_transform(transform_name, args)
            for transform_name, args in transform_dict.items()
        ]
    )


def read_dicoms_from_disk(
    config: dict,
    verbose: bool = False,
) -> tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.Subject]:
    """
    Get all the data used in the training process - training, validation and testing
    This reads in the DICOMs created by `scripts/create_dicoms.py`.
    Transforms are applied as defined in the configuration (see userconf.yml).
    Prints a progress bar.

    :param config: The configuration, e.g. from userconf.yml
    :param verbose: whether to print extra stuff, in case we want to be sure about where
                    we're reading from

    :returns: subjects for training
    :returns: subjects for validation
    :returns: a subject, for testing

    :raises: ValueError if transforms is not "default", "none" or a tio.transforms.Transform

    """
    # Read in data + convert to subjects
    window_size = transform.window_size(config)
    train_subjects, test_subjects, val_subjects = (
        [
            subject(path, window_size)
            for path in tqdm(
                files.dicom_paths(config, mode, verbose), desc=f"Reading {mode} DICOMs"
            )
        ]
        for mode in ("train", "test", "val")
    )

    # Convert to SubjectsDatasets, which is where the transforms get applied

    train_subjects = tio.SubjectsDataset(
        train_subjects, transform=_transforms(config["transforms"])
    )
    val_subjects = tio.SubjectsDataset(val_subjects)
    (test_subject,) = test_subjects

    return train_subjects, val_subjects, test_subject


def get_patch_size(config: dict[str, Any]) -> tuple[int, int, int]:
    """
    Get the patch size from the configuration

    :param config: The configuration, that might be from userconf.yml
    :returns: The patch size as a tuple of ints ZYX

    """
    return tuple(int(x) for x in config["patch_size"].split(","))
