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
    label_path = pathlib.Path(args.label_path)
    labels = tifffile.imread(label_path)

    # Find the unique labels
    unique_labels = np.unique(labels)

    # Define which colours to use
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(cmap.N)]
    hex_colors = [rgb2hex(color) for color in colors]

    colours = np.empty(labels.shape, dtype=object)
    for label in unique_labels:
        colours[labels == label] = hex_colors[label]

    # Plot the voxels
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), subplot_kw={"projection": "3d"})
    pbar = tqdm(total=3 * (len(unique_labels) - 1))
    for axis, elev, azim in zip(axes, [0, 0, 90], [0, 90, 90]):
        for label in unique_labels:
            # Don't plot the background
            if not label:
                continue

            # Find co-ords
            co_ords = np.argwhere(labels == label)

            axis.scatter(
                co_ords[:, 0],
                co_ords[:, 1],
                co_ords[:, 2],
                c=hex_colors[label],
                s=2,
                label=label,
            )

            pbar.update(1)

        axis.view_init(elev=elev, azim=azim)

        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])

    axes[2].legend()

    fig.suptitle(label_path.stem)
    fig.tight_layout()

    out_path = (
        files.boring_script_out_dir() / "voxels" / f"{label_path.stem}_voxels.png"
    )
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)
    fig.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the provided TIFF of labels as voxels"
    )
    parser.add_argument("label_path", type=str, help="Path to the TIFF of labels")

    main(parser.parse_args())
