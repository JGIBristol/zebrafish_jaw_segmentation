"""
Plot stuff
"""

import torch
import matplotlib.pyplot as plt


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
        axis.imshow(heatmap_slice, cmap="afmhot_r", alpha=0.3)

        axis.axis("off")

    return fig, axes
