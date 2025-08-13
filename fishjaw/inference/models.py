"""
Get and use the different models for inference
"""

import pathlib
import functools

import torch
import numpy as np

from ..localisation.model import get_model, crop


@functools.cache
def get_jaw_loc_model(*, device: str) -> torch.nn.Module:
    """
    Get the network used to locate the jaw in a CT scan

    Returned in evaluation mode - i.e. suitable for inference but not
    training.

    :param device: "cuda" to run on GPU, else "cpu"

    :returns: the trained model for locating the jaw
    """
    model = get_model(device)
    path = pathlib.Path(__file__).parents[2] / "model" / "jaw_loc_model.pth"
    with open(path, "rb") as f:
        model.load_state_dict(torch.load(f))

    model.eval()

    return model


@functools.cache
def get_jaw_segment_model() -> torch.nn.Module:
    """
    Get the network used to segment the jaw from a cropped CT scan
    """


def crop_jaw(ct_scan: np.ndarray, jaw_loc_model: torch.nn.Module) -> np.ndarray:
    """
    Crop the jaw region from the CT scan using the model.
    """


def segment_jaw(
    cropped_ct_scan: np.ndarray, jaw_segment_model: torch.nn.Module
) -> np.ndarray:
    """
    Segment the jaw from a cropped CT scan
    """
