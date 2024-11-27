"""
Plotting 3D things

"""

import torch
import numpy as np
from numpy.typing import NDArray
import torchio as tio
import matplotlib
import matplotlib.pyplot as plt

from ..model import model


def plot_slices(
    arr: NDArray[np.float32], mask: NDArray[np.int8] | None = None
) -> tuple[matplotlib.figure.Figure, NDArray[matplotlib.axes.Axes]]:
    """
    Plot slices of a 3d array

    """
    if mask is not None:
        if mask.shape != arr.shape:
            raise ValueError("Array and mask must have the same shape")

    indices = np.floor(np.arange(0, arr.shape[0], arr.shape[0] // 16)).astype(int)
    vmin, vmax = np.min(arr), np.max(arr)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in zip(indices, axes.flat):
        ax.imshow(arr[i], cmap="gray", vmin=vmin, vmax=vmax)
        if mask is not None:
            ax.imshow(mask[i], cmap="hot_r", alpha=0.5)
        ax.axis("off")
        ax.set_title(i)

    fig.tight_layout()

    return fig, axes


def plot_subject(
    subject: tio.Subject,
) -> tuple[matplotlib.figure.Figure, NDArray[matplotlib.axes.Axes]]:
    """
    Plot the image and label of a subject

    """
    image = subject[tio.IMAGE][tio.DATA].squeeze().numpy()
    label = subject[tio.LABEL][tio.DATA].squeeze().numpy()

    return plot_slices(image, label)


def plot_inference(
    net: torch.nn.Module,
    subject: tio.Subject,
    *,
    patch_size: tuple[int, int, int],
    patch_overlap: tuple[int, int, int],
    activation: str = "softmax",
) -> matplotlib.figure.Figure:
    """
    Plot the inference on an image

    Helper function that just bundles together `fishjaw.model.predict`
    and `fishjaw.visualisation.images_3d.plot_slices`

    """
    assert activation in {"softmax", "sigmoid"}

    # Perform inference
    prediction = model.predict(
        net,
        subject,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        activation=activation,
    )

    # Get the image from the subject
    image = subject[tio.IMAGE][tio.DATA].squeeze().numpy()

    fig, _ = plot_slices(image, prediction)
    fig.suptitle("Model Prediction")
    fig.tight_layout()

    return fig
