"""
Copy TIFF files from the DATABASE/ on the RDSF to a subfolder in Felix + Rich's directory.

Just so I don't break anything in the original data, I've been working off TIFF files in
1Felix and Rich make models/wahabs_scans/
This script:
 1. Copies the existing 3D TIFF files from the database to this directory
 2. Converts the remaining 2D TIFFs to 3D, renames them from the old_n to n numbering scheme
    and saves them here

"""

import argparse
import pathlib

import tqdm
import tifffile
import numpy as np

from fishjaw.util import util


def create_3d_tiff(input_dir: pathlib.Path, output_file: pathlib.Path):
    """
    Write a 3D tiff to the provided path, by stacking the 2D tiffs in the provided directory.

    """
    if output_file.exists():
        print(f"Output file {output_file} already exists, skipping")
        return

    # Get a sorted list of all TIFF files in the input directory
    tiff_files = sorted([f for f in input_dir.glob("*.tiff")])

    # Read each image and append it to the stack
    stack = []
    for tiff_file in tqdm.tqdm(tiff_files):
        image = tifffile.imread(tiff_file)
        stack.append(image)

    # Convert the stack to a 3D numpy array
    stack = np.stack(stack, axis=0)

    # Save the stack as a 3D TIFF
    tifffile.imwrite(output_file, stack)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(help=__doc__)
    parser.parse_args()

    # Read the metadata mastersheet so we can map from old_n to new n
    # For all the 3D tiffs, in the mastersheet copy 3D tifs over
    # For all the 2D tiffs that don't already exist, convert 2D tifs to 3D and save them

