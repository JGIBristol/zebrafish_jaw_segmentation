"""
Plot the specified 3D tiff of labels as voxels

"""

import pathlib
import argparse

import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from fishjaw.util import files


def main(args: argparse.Namespace):
    """
    Read the label file, find the unique labels, plot colour-coded voxels

    """
    # Read the labels
    labels = tifffile.imread(args.label_path)[::4, ::4, ::4]

    # Find the unique labels
    unique_labels = np.unique(labels)

    # Define which colours to use
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(cmap.N)]
    hex_colors = [rgb2hex(color) for color in colors]

    colours = np.empty(labels.shape, dtype=object)
    for label in unique_labels:
        # Ignore the background
        if not label:
            continue

        colours[labels == label] = hex_colors[label]

    # Plot the voxels
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), subplot_kw={"projection": "3d"})
    for axis, elev, azim in tqdm(zip(axes, [0, 90, 0], [0, 0, 90]), total=3):
        axis.voxels(labels, facecolors=colours, edgecolors="k")
        axis.view_init(elev=elev, azim=azim)

        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])

    path = (
        files.boring_script_out_dir()
        / f"{pathlib.Path(args.label_path).stem}/_voxels.png"
    )
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the provided TIFF of labels as voxels"
    )
    parser.add_argument("label_path", type=str, help="Path to the TIFF of labels")

    main(parser.parse_args())
