"""
Plot slices of the inferences performed in `segment_jaws.py`
"""

import argparse
import tifffile

from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.inference import read
from fishjaw.visualisation import images_3d


def main():
    """
    Read in pairs of images and masks, plot slices and save them
    """
    in_dir = files.script_out_dir() / "jaw_segmentations"
    img_in_dir = in_dir / "imgs"
    mask_in_dir = in_dir / "masks"

    out_dir = in_dir / "slices"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = sorted(list(img_in_dir.glob("*.tif")))
    in_masks = sorted(list(mask_in_dir.glob("*.tif")))
    for img, mask in tqdm(zip(in_imgs, in_masks, strict=True), total=len(in_imgs)):
        metadata = read.metadata(read.fish_number(img))

        i = tifffile.imread(img)
        m = tifffile.imread(mask)

        fig, _ = images_3d.plot_slices(i, m)
        fig.suptitle(str(metadata))
        fig.tight_layout()

        fig.savefig(out_dir / img.name.replace(".tif", ".png"))
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    main(**vars(args))
