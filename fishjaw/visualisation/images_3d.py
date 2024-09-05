"""
Plotting 3D things

"""

import matplotlib.pyplot as plt

import numpy as np


def plot_slices(
    arr: np.ndarray, mask: np.ndarray = None
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
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
