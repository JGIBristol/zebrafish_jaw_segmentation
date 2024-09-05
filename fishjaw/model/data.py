"""
Loading, pre-processing, etc. the data for the model

"""

import pathlib

import torch
import numpy as np
import torchio as tio

from ..images import io, transform


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
    centre: tuple[int, int, int] = None,
) -> tio.Subject:
    """
    Create a subject from a DICOM file
    Optionally, crop the image with the provided centre
    and the size defined in the userconf file

    :param dicom_path: Path to the DICOM file
    :param centre: The centre of the window.

    :returns: The subject

    """
    image, mask = io.read_dicom(dicom_path)

    if centre is not None:
        image = transform.crop(image, centre)
        mask = transform.crop(mask, centre)

    # Need to copy since torch doesn't support non-writable tensors
    # Convert to a float in [0, 1]
    image = image.copy() / 65535
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
    shuffle: bool,
    patch_size: tuple[int, int, int],
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader from a SubjectsDataset

    The training data should be shuffled; validation data should not be

    :param subjects: The dataset. Training data should have random transforms applied
    :param shuffle: Whether to shuffle the data
    :param patch_size: The size of the patches to extract
    :param batch_size: The batch size

    :returns: The loader

    """
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
