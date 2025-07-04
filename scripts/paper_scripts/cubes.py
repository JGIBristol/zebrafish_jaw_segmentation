import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.metrics import hausdorff_distance


def dice(im1: np.ndarray, im2: np.ndarray) -> float:
    """Calculate Dice score for 3D volumes."""
    intersection = np.sum(im1 * im2)

    return (2.0 * intersection) / (np.sum(im1) + np.sum(im2))


def hausdorff_score(im1: np.ndarray, im2: np.ndarray) -> float:
    """Calculate the Hausdorff score for 3D volumes."""
    max_dist = np.sqrt(im1.shape[0] ** 2 + im1.shape[1] ** 2 + im1.shape[2] ** 2)
    return 1 - (hausdorff_distance(im1, im2) / max_dist)


def combined_hausdorff_dice_metric(
    im1: np.ndarray, im2: np.ndarray, alpha: float
) -> float:
    """Calculate the combined Hausdorff-Dice metric."""
    return alpha * dice(im1, im2) + (1 - alpha) * hausdorff_score(im1, im2)


def create_cube(dim):
    """Create a 3D binary cube volume."""
    size, offset = 20, 15
    start, end = offset, offset + size

    cube = np.zeros(dim)
    cube[start:end, start:end, start:end] = 1

    return cube


def add_protrusions(cube):
    """Add non-symmetrical sticky-outy bits to the cube."""
    cube[45:50, 10:15, 10:20] = 1
    return cube


def create_cubes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = (50, 50, 50)
    cube1 = create_cube(dim)

    # Distorted non-symmetrical cube with protrusions
    cube2 = create_cube(dim)
    cube2 = add_protrusions(cube2)
    cube2[cube2 > 0.5] = 1
    cube2[cube2 <= 0.5] = 0

    # Symmetrical cube with distortion (blurred cube)
    cube3 = create_cube(dim)
    cube3 = gaussian_filter(cube3, sigma=4)
    cube3[cube3 > 0.5] = 1
    cube3[cube3 <= 0.5] = 0

    return cube1, cube2, cube3


def main(alpha: float):
    """
    Create the cubes/deformed cubes, calculate the metrics, create axes and plot
    """
    cube1, cube2, cube3 = create_cubes()

    dice_scores = [dice(cube1, cube) for cube in (cube1, cube2, cube3)]
    hausdorff_scores = [hausdorff_score(cube1, cube) for cube in (cube1, cube2, cube3)]
    combined_scores = [
        combined_hausdorff_dice_metric(cube1, cube, alpha)
        for cube in (cube1, cube2, cube3)
    ]

    scores = pd.DataFrame(
        {
            "Cube": ["Cube 1", "Cube 2", "Cube 3"],
            "Dice Score": dice_scores,
            "Hausdorff Score": hausdorff_scores,
            "Combined Score": combined_scores,
        }
    )

    # Visualization
    fig, (cube_axes, table_axes) = plt.subplots(
        2, 3, figsize=(15, 10), subplot_kw={"projection": "3d"}
    )
    cube_kw = {"edgecolor": "none", "alpha": 0.6}

    for axis, cube, colour in zip(
        cube_axes, (cube1, cube2, cube3), ("blue", "red", "green")
    ):
        axis.voxels(cube, facecolors=colour, **cube_kw)
        axis.set_xlim(0, 50)
        axis.set_ylim(0, 50)
        axis.set_zlim(0, 50)

    plt.tight_layout()

    # Add tables to the axes
    for axis, (_, row) in zip(table_axes.flatten(), scores.iterrows()):
        print(row[1:])
        axis.axis("off")

        # Convert the row data to the format expected by plt.table
        cell_data = [
            [f"{val:.3f}" if isinstance(val, float) else str(val)] for val in row[1:]
        ]
        col_labels = scores.columns[1:].tolist()

        table = axis.table(
            cellText=cell_data,
            rowLabels=col_labels,
            cellLoc="center",
            loc="best",
            colWidths=[0.3],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(1, 4)  # Make rows taller

        axis.set_xlim(0, 1)

    image_filename = "3d_cube_comparison_dice_hausdorff.png"
    plt.savefig(image_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as '{image_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Cube Comparison with Dice and Hausdorff Metrics"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Weighting factor for the combined Hausdorff-Dice metric",
    )
    main(**vars(parser.parse_args()))
