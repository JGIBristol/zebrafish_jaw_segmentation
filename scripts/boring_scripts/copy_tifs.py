"""
Copy TIFF files from the DATABASE/ on the RDSF to a subfolder in Felix + Rich's directory.

Just so I don't break anything in the original data, I've been working off TIFF files in
1Felix and Rich make models/wahabs_scans/
This script:
 1. Copies the existing 3D TIFF files from the database to this directory
 2. Converts the remaining 2D TIFFs to 3D, renames them from the old_n to n numbering scheme
    and saves them here

"""

import os
import sys
import shutil
import pathlib
import warnings
import argparse

from tqdm import tqdm
import tifffile
import numpy as np

from fishjaw.util.files import _mastersheet, rdsf_dir
from fishjaw.util.util import userconf


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    # Read the metadata mastersheet so we can map from old_n to new n
    metadata = _mastersheet()[["n", "old_n"]]
    mapping = {int(row.old_n): int(row.n) for _, row in metadata.iterrows()}

    rdsf_dir_ = rdsf_dir(userconf())
    out_dir = rdsf_dir_ / "1Felix and Rich make models" / "wahabs_scans"

    # For all the 3D tiffs, in the mastersheet copy 3D tifs over
    database_dir = rdsf_dir_ / "DATABASE" / "uCT" / "Wahab_clean_dataset"
    wahab_3d_tif_dir = database_dir / "TIFS/"

    pbar = tqdm(list(wahab_3d_tif_dir.glob("ak_*.tif")))
    for img_path in pbar:
        n = int(img_path.stem.split("ak_")[1])

        output_img = out_dir / f"{n}.tif"
        if output_img.exists():
            continue

        pbar.set_description(f"Copying to {output_img.name}")
        try:
            shutil.copyfile(img_path, output_img)
        except KeyboardInterrupt as e:
            warnings.warn(
                f"removing partially copied {output_img}, {os.path.getsize(output_img) // (1024 * 1024)} MB"
            )
            os.remove(output_img)
            raise e
        except OSError as e:
            print(f"Problem for {n}, skipping: {str(e)}", file=sys.stderr)

    # For all the 2D tiffs that don't already exist, convert 2D tifs to 3D and save them
    wahab_2d_tif_dir = database_dir / "low_res_clean_v3"
    pbar = tqdm(list(wahab_2d_tif_dir.glob(r"[0-9][0-9][0-9]")))
    for dir_ in pbar:
        old_n = int(dir_.name)
        try:
            new_n = mapping[old_n]
        except KeyError:
            print(f"No metadata found for old_n {old_n}, skipping", file=sys.stderr)
            continue

        output_img = out_dir / f"{new_n}.tif"
        if output_img.exists():
            continue

        pbar.set_description(f"Creating {output_img.name} from {old_n}")
        create_3d_tiff(dir_, output_img)
