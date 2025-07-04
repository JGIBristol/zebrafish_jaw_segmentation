import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.metrics import hausdorff_distance


def dice_coefficient_3d(im1, im2):
    """Calculate Dice score for 3D volumes."""
    intersection = np.sum(im1 * im2)

    return 2.0 * intersection / (np.sum(im1) + np.sum(im2))


def create_cube(dim, size, offset=(0, 0, 0)):
    """Create a 3D binary cube volume."""

    cube = np.zeros(dim)

    x_start, x_end = offset[0], offset[0] + size

    y_start, y_end = offset[1], offset[1] + size

    z_start, z_end = offset[2], offset[2] + size

    cube[x_start:x_end, y_start:y_end, z_start:z_end] = 1

    return cube


def add_protrusions(cube):
    """Add non-symmetrical sticky-outy bits to the cube."""

    cube[30:40, 35:40, 30:35] = 1  # Protrusion on one corner

    cube[10:20, 5:10, 40:45] = 1  # Protrusion on another side

    cube[40:45, 10:15, 10:20] = 1  # Another irregular protrusion

    return cube


def plot_cube(cube, ax, color="blue", alpha=0.3):
    """Plot a cube from a 3D binary volume."""

    ax.voxels(cube, facecolors=color, edgecolor="k", alpha=alpha)


def combined_hausdorff_dice_metric(dice_score, hausdorff_dist, max_dist):
    """Calculate the combined Hausdorff-Dice metric."""

    normalized_hausdorff = hausdorff_dist / max_dist

    combined_metric = 0.5 * dice_score + 0.5 * (1 - normalized_hausdorff)

    return combined_metric


def main():
    """
    Create the cubes/deformed cubes, calculate the metrics, create axes and plot
    """
    # Create three cubes: a perfect cube, a distorted one, and a blurred symmetrical cube
    dim = (50, 50, 50)
    cube1 = create_cube(dim, size=20, offset=(15, 15, 15))

    # Distorted non-symmetrical cube with protrusions
    cube2 = create_cube(dim, size=20, offset=(15, 15, 15))
    cube2 = add_protrusions(cube2)
    cube2 = gaussian_filter(cube2, sigma=1)
    cube2[cube2 > 0.5] = 1
    cube2[cube2 <= 0.5] = 0

    # Symmetrical cube with distortion (blurred cube)
    cube3 = create_cube(dim, size=20, offset=(15, 15, 15))
    cube3 = gaussian_filter(cube3, sigma=4)
    cube3[cube3 > 0.5] = 1
    cube3[cube3 <= 0.5] = 0

    dice_score1 = dice_coefficient_3d(cube1, cube1)
    dice_score2 = dice_coefficient_3d(cube1, cube2)
    dice_score3 = dice_coefficient_3d(cube1, cube3)

    # Calculate Hausdorff distances
    hausdorff_dist1 = hausdorff_distance(cube1, cube1)
    hausdorff_dist2 = hausdorff_distance(cube1, cube2)
    hausdorff_dist3 = hausdorff_distance(cube1, cube3)

    # Calculate maximum possible distance (diagonal of the volume's bounding box)
    max_dist = np.sqrt(dim[0] ** 2 + dim[1] ** 2 + dim[2] ** 2)

    # Calculate combined Hausdorff-Dice metrics
    combined_metric1 = combined_hausdorff_dice_metric(
        dice_score1, hausdorff_dist1, max_dist
    )
    combined_metric2 = combined_hausdorff_dice_metric(
        dice_score2, hausdorff_dist2, max_dist
    )
    combined_metric3 = combined_hausdorff_dice_metric(
        dice_score3, hausdorff_dist3, max_dist
    )

    # Visualization
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    plot_cube(cube1, ax1, color="blue", alpha=0.6)
    ax1.set_title(
        f"Cube 1 (Perfect)¥nDice: {dice_score1:.2f}, Hausdorff: {hausdorff_dist1:.2f}¥nCombined: {combined_metric1:.2f}"
    )

    ax2 = fig.add_subplot(132, projection="3d")
    plot_cube(cube2, ax2, color="red", alpha=0.6)
    ax2.set_title(
        f"Cube 2 (Non-Symmetrical)¥nDice: {dice_score2:.2f}, Hausdorff: {hausdorff_dist2:.2f}¥nCombined: {combined_metric2:.2f}"
    )

    ax3 = fig.add_subplot(133, projection="3d")
    plot_cube(cube3, ax3, color="green", alpha=0.6)
    ax3.set_title(
        f"Cube 3 (Symmetrical Blur)¥nDice: {dice_score3:.2f}, Hausdorff: {hausdorff_dist3:.2f}¥nCombined: {combined_metric3:.2f}"
    )
    plt.tight_layout()

    image_filename = "3d_cube_comparison_dice_hausdorff.png"
    plt.savefig(image_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as '{image_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Cube Comparison with Dice and Hausdorff Metrics")
    main(**vars(parser.parse_args()))