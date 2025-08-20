"""
Read in full-size CT scans from Wahab's 3D TIF directory, locate the
jaw and segment it out

Saves the cropped jaw TIF and segmentation mask
"""

import sys
import pathlib
import argparse
import tifffile

from tqdm import tqdm

from fishjaw.util import files, util
from fishjaw.inference import models
from fishjaw.images.transform import CropOutOfBoundsError


def main(crop_size: int):
    """
    Get a list of the files in Wahab's dir, load in the two models,
    iterate over the files cropping + performing inference

    Then save each for later analysis
    """
    out_dir = files.script_out_dir() / "jaw_segmentations"

    img_out_dir = out_dir / "imgs"
    mask_out_dir = out_dir / "masks"

    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # Get the input dir
    config = util.userconf()
    input_dir = (
        pathlib.Path(config["rdsf_dir"])
        / "DATABASE"
        / "uCT"
        / "Wahab_clean_dataset"
        / "TIFS"
    )

    # Get the models
    device = config["device"]
    loc_model = models.get_jaw_loc_model(device=device)
    seg_model = models.get_jaw_segment_model(device=device)

    for img_path in tqdm(sorted(list(input_dir.glob("*.tif")))):
        name = img_path.name
        if (img_out_dir / name).exists() and (mask_out_dir / name).exists():
            print(f"Skipping {name}")
            continue

        try:
            scan = tifffile.imread(img_path)
        except ValueError as e:
            print(
                f"Error reading {name}; is the tiff file incomplete?\n{str(e)}",
                file=sys.stderr,
            )
            continue


        # Crop the image
        try:
            cropped = models.crop_jaw(
                loc_model, scan, window_size=tuple([crop_size] * 3)
            )
        except CropOutOfBoundsError as e:
            print(
                f"Error cropping {name}; likely an issue with the jaw localising model\n{str(e)}",
                file=sys.stderr,
            )
            continue

        # Run inference
        prediction = models.segment_jaw(cropped, seg_model)

        # Save
        tifffile.imwrite(img_out_dir / name, cropped)
        tifffile.imwrite(mask_out_dir / name, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=192,
        help="Size of region (in px) to crop around the predicted jaw centre",
    )

    args = parser.parse_args()

    main(**vars(args))
