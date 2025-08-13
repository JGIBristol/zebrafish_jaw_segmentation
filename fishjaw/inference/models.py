"""
Get and use the different models for inference

This should be the only interface needed to interact with the models;
some of the functions here are just thin wrappers around other helper
functions, they're just here to keep everything in one place
"""

import pathlib
import functools

import torch
import numpy as np
import torchio as tio

from ..localisation.model import get_model, crop
from ..model import data
from ..model.model import load_model, predict
from ..images.metrics import largest_connected_component


class InferenceError(Exception):
    """
    Something went wrong
    """


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
def get_jaw_segment_model(device: str) -> torch.nn.Module:
    """
    Get the network used to segment the jaw from a cropped CT scan

    :param device: either "cuda" to run on GPU or "cpu"
    :returns: trained jaw segmentation model
    """
    return load_model("paper_model.pkl").load_model(set_eval=True).to(device)


def crop_jaw(
    jaw_loc_model: torch.nn.Module,
    ct_scan: np.ndarray,
    *,
    window_size: tuple[int, int, int],
) -> np.ndarray:
    """
    Crop the jaw region from the CT scan using the model.

    Performs inference on the device that the model is on - this might
    give unexpected results if the model is on multiple devices.

    :param jaw_loc_model: the model used to locate the jaw in a CT scan
    :param ct_scan: 3D greyscale numpy array
    :param window_size: size of the returned cropped image

    :returns: 3D numpy array of the cropped image
    """
    # This is the size that i downsampled the input images to when i trained the model
    # It's hard-coded here, which isn't ideal, but it should be fine since
    # I don't expect this to change (unless we train another model which has
    # a different input size, in which case we will need to change it)
    model_input_size = (512, 128, 128)

    return crop(
        jaw_loc_model,
        ct_scan,
        model_input_size=model_input_size,
        window_size=window_size,
    )


def segment_jaw(
    cropped_ct_scan: np.ndarray,
    jaw_segment_model: torch.nn.Module,
    *,
    threshold: float | None = 0.5,
    largest_component: bool = True,
) -> np.ndarray:
    """
    Segment the jaw from a cropped CT scan

    Thresholds the model's output at 0.5, and takes the largest connected component

    :param cropped_ct_scan: 3D CT scan to perform inference on
    :param jaw_segment_model: trained model for performing jaw segmentation
    :param threshold: either a float, in which case the output is thresholded
    """
    if largest_component and threshold is None:
        raise InferenceError(
            "Cannot take the largest connected component unless model output is thresholded"
        )

    # These are hard-coded because they are the settings we trained the
    # model with
    patch_size = (160, 160, 160)
    patch_overlap = (4, 4, 4)
    activation = "sigmoid"

    scaled = data.ints2float(cropped_ct_scan)
    tensor = torch.as_tensor(scaled, dtype=torch.float32).unsqueeze(0)
    subject = tio.Subject(image=tio.Image(tensor=tensor, type=tio.INTENSITY))
    prediction = predict(
        jaw_segment_model,
        subject,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        activation=activation,
    )

    if threshold is not None:
        prediction = prediction > threshold

    if largest_component:
        prediction = largest_connected_component(prediction)

    return prediction
