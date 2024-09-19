"""
Loading, pre-processing, etc. the data for the model

"""

import pathlib
from typing import Union

import torch
import torch.utils
import numpy as np
import torchio as tio
from tqdm import tqdm

from ..images import io, transform
from ..util import files


def ints2float(int_arr: np.ndarray) -> np.ndarray:
    """
    Scale an array from 16-bit integer values to float values in [0, 1]

    :param int_arr: The array to scale. Should be 16-bit integers - might be stored as a 32-bit
                    datatype, but the values should be in the range of a 16-bit integer

    :returns: The scaled array, with values between 0 and 1

    :raises: ValueError if the max value is less than the max value of a uint8 or greater than the max value of a uint16
    :raises: ValueError if the array is not of integer type

    """
    # Check that the values are in the right range
    if (max_val := int_arr.max()) < np.iinfo(np.uint8).max:
        raise ValueError(
            f"Max value {max_val} is less than can be stored by a uint8 - are you sure this is 16-bit data?"
        )
    elif max_val > (uint16max := np.iinfo(np.uint16).max):
        raise ValueError(
            f"Max value {max_val} is greater than can be stored by a uint16 - are you sure this is 16-bit data?"
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


def subject(
    dicom_path: pathlib.Path,
    *,
    centre: tuple[int, int, int] = None,  # Shadowing whoops
    window_size: tuple[int, int, int] = None,
) -> tio.Subject:
    """
    Create a subject from a DICOM file
    Optionally, crop the image with the provided centre and window_size

    :param dicom_path: Path to the DICOM file
    :param centre: The centre of the window.

    :returns: The subject
    :raises: ValueError if exactly one of centre and window_size specified

    """
    if (centre is None) != (window_size is None):
        raise ValueError(
            f"Must specify either both or neither centre ({centre}) and window_size ({window_size})"
        )

    image, mask = io.read_dicom(dicom_path)

    if centre is not None:
        # We know implicitly that window_size is not None
        image = transform.crop(image, centre, window_size)
        mask = transform.crop(mask, centre, window_size)

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


def train_val_loader(
    subjects: tio.SubjectsDataset,
    *,
    train: bool,
    patch_size: tuple[int, int, int],
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader from a SubjectsDataset

    Training data is shuffled and has the last batch dropped; validation data is not

    :param subjects: The dataset. Training data should have random transforms applied
    :param train: If we're training or not
    :param patch_size: The size of the patches to extract
    :param batch_size: The batch size

    :returns: The loader

    """
    shuffle = train is True
    drop_last = train is True

    # Even probability of the patches being centred on each value
    label_probs = {0: 1, 1: 1}
    patch_sampler = tio.LabelSampler(
        patch_size=patch_size, label_probabilities=label_probs
    )

    patches = tio.Queue(
        subjects,
        max_length=10000,  # Not sure if this matters
        samples_per_volume=1,
        sampler=patch_sampler,
        num_workers=0,
        shuffle_patches=True,
        shuffle_subjects=True,
    )

    return torch.utils.data.DataLoader(
        patches,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Load the data in the main process
        pin_memory=False,  # No idea why I have to set this to False, otherwise we get obscure errors
        drop_last=drop_last,
    )


def test_loader(
    patches: tio.GridSampler, *, batch_size: int
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for testing - this samples patches from the image in a grid
    such that we can reconstruct the entire image

    :param patches: patches from a subject, created with e.g. tio.GridSampler(subject, patch_size, patch_overlap=(4, 4, 4))
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


def _transforms() -> tio.transforms.Transform:
    """
    Define the transforms to apply to the training data

    """
    return tio.Compose(
        [
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            tio.RandomAffine(
                p=0.25,
                degrees=10,
                scales=0.2,
            ),
            # tio.RandomBlur(p=0.3),
            # tio.RandomBiasField(0.4, p=0.5),
            # tio.RandomNoise(0.1, 0.01, p=0.25),
            # tio.RandomGamma((-0.3, 0.3), p=0.25),
            # tio.ZNormalization(),
            # tio.RescaleIntensity(percentiles=(0.5, 99.5)),
        ]
    )


def patch_size(config: dict) -> tuple[int, int, int]:
    """
    Get the patch size from the configuration

    :param config: The configuration, that might be from userconf.yml
    :returns: The patch size as a tuple of ints ZYX

    """
    return tuple(int(x) for x in config["patch_size"].split(","))


def centre(dicom_path: pathlib.Path) -> tuple[int, int, int]:
    """
    Get the centre of the jaw for a given fish

    """
    n = int(dicom_path.stem.split("_", maxsplit=1)[-1])
    return transform.centre(n)


def get_data(
    config: dict,
    rng: np.random.Generator,
    *,
    train_frac: float = 0.95,
    transforms: Union[str | tio.transforms.Transform] = "default",
) -> tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.Subject]:
    """
    Get all the data used in the training process - training, validation and testing
    This reads in the DICOMs created by `scripts/create_dicoms.py`
    Prints a progress bar

    :param config: The configuration, e.g. from userconf.yml
    :param rng: A random number generator to use for test/train/split
    :param train_frac: The fraction of the data to use for training (roughly)
    :param transforms: whether to apply transforms to the training data.
                       "default" applies the transforms in transforms();
                       "none" applies no transforms;
                       otherwise applies the transforms provided

    :return: subjects for training
    :return: subjects for validation
    :return: a subject, for testing

    :raises: ValueError if transforms is not "default", "none" or a tio.transforms.Transform

    """
    # Choose the transforms
    if isinstance(transforms, (str, tio.transforms.Transform)):
        if transforms == "default":
            transforms = _transforms()
        elif transforms == "none":
            # This will apply no transforms when passed to the SubjectsDataset
            transforms = None
    else:
        raise ValueError(
            f"transforms must be 'default', 'none' or a tio.transforms.Transform, not {transforms}"
        )

    # Read in data + convert to subjects
    dicom_paths = sorted(list(files.dicom_dir(config).glob("*.dcm")))
    subjects = [
        subject(path, centre=centre(path), window_size=transform.window_size(config))
        for path in tqdm(dicom_paths, desc="Reading DICOMs")
    ]

    # Choose some indices to act as train, validation and test
    # This is a bit of a hack
    indices = np.arange(len(subjects))
    rng.shuffle(indices)
    train_idx, val_idx, test_idx = np.split(
        indices, [int(train_frac * len(indices)), len(indices) - 1]
    )
    assert len(test_idx) == 1
    test_idx = test_idx[0]

    print(f"Train: {len(train_idx)=}")
    print(f"Val: {val_idx=}")
    print(f"Test: {test_idx=}")

    train_subjects = tio.SubjectsDataset(
        [subjects[i] for i in train_idx], transform=_transforms()
    )
    val_subjects = tio.SubjectsDataset([subjects[i] for i in val_idx])
    test_subject = subjects[test_idx]

    return train_subjects, val_subjects, test_subject
