"""
Plot stuff
"""

import torch
from matplotlib import colors
import matplotlib.pyplot as plt


def _transparent_cmap():
    """
    Range from transparent black to white
    """
    c_black = colors.colorConverter.to_rgba("black", alpha=0)
    c_red = colors.colorConverter.to_rgba("red", alpha=1)
    return colors.LinearSegmentedColormap.from_list(
        "heatmap_cmap", [c_black, c_red], N=256
    )


def plot_heatmap(
    img: torch.Tensor, heatmap: torch.Tensor
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """
    Plot three slices of the heatmap through different slices

    The tensors should be on the CPU
    """
    # Find the index of the Z slice where the centroid has the highest sum

    fig, axes = plt.subplot_mosaic(
        """
                             AAAB
                             AAAB
                             AAAB
                             CCC.
                             """,
        figsize=(6, 6),
    )

    for axis, permutation in zip(
        [axes["A"], axes["B"], axes["C"]],
        [(0, 1, 2, 3, 4), (0, 1, 4, 2, 3), (0, 1, 3, 4, 2)],
    ):
        permuted_img = img.permute(permutation)
        permuted_heatmap = heatmap.permute(permutation)

        centre = torch.argmax(permuted_heatmap[0][0].sum(dim=(1, 2))).item()

        img_slice = permuted_img[0][0][centre].numpy()
        heatmap_slice = permuted_heatmap[0][0][centre].numpy()

        axis.imshow(img_slice, cmap="gray")
        im = axis.imshow(heatmap_slice, cmap=_transparent_cmap())

        axis.axis("off")

    fig.colorbar(im, ax=axes["C"], orientation="vertical", fraction=0.046, pad=0.04)

    return fig, axes


def plot_centroid(
    img: torch.tensor, centroid: tuple[int, int, int]
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """
    Plot the predicted centroid on the image.

    """
    fig, axes = plt.subplot_mosaic(
        """
                             AAAB
                             AAAB
                             AAAB
                             CCC.
                             """,
        figsize=(6, 6),
    )

    for axis, permutation in zip(
        [axes["A"], axes["B"], axes["C"]],
        [(0, 1, 2, 3, 4), (0, 1, 4, 2, 3), (0, 1, 3, 4, 2)],
    ):
        permuted_img = img.permute(permutation)

        # Take the right slice
        img_slice = permuted_img[0][0][centroid[permutation[2] - 2]].numpy()
        centroid_x, centroid_y = (
            centroid[permutation[3] - 2],
            centroid[permutation[4] - 2],
        )

        axis.imshow(img_slice, cmap="gray")
        axis.scatter(centroid_y, centroid_x, color="red", s=10, marker="x")

        axis.axis("off")

    return fig, axes
