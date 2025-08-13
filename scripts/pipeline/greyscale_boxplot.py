"""
Plot boxplots of the greyscale distribution of the jaws
"""

import argparse
import tifffile

from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.visualisation import images_3d


def main():
    """
    Read in the mastersheet to get metadata from the different segmentations

    Then read in pairs of images and masks
    """
    mastersheet = files._mastersheet()
    mastersheet.set_index("n")
    print(mastersheet.to_markdown())

    in_dir = files.script_out_dir() / "jaw_segmentations"
    img_in_dir = in_dir / "imgs"
    mask_in_dir = in_dir / "masks"

    out_dir = in_dir / "boxplot"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = sorted(list(img_in_dir.glob("*.tif")))
    in_masks = sorted(list(mask_in_dir.glob("*.tif")))

    for img, mask in tqdm(zip(in_imgs, in_masks, strict=True), total=len(in_imgs)):
        i = tifffile.imread(img)
        m = tifffile.imread(mask)

        greyscale_vals = i[m]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    main(**vars(args))
