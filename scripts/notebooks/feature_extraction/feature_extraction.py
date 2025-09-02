"""
Extract radiomic features from medical images + masks in a provided directory

NB this script uses a standalone python env, since it relies on the pyradiomics
package which is incompatible with the version of python that I've been using otherwise.

The environment is specified in the pyproject.toml file in the same directory as this file;
you can activate it by running "uv run scripts/notebooks/feature_extraction/feature_extraction.py"
as normal - uv will handle creating and activating the environment for you.

"""

import argparse

import tifffile
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.notebook import tqdm
from radiomics import featureextractor

from fishjaw.util import files
from fishjaw.inference import read


def main():
    in_dir = files.script_out_dir() / "jaw_segmentations"
    img_in_dir = in_dir / "imgs"
    mask_in_dir = in_dir / "masks"

    img_paths = sorted(list(img_in_dir.glob("*.tif")))
    mask_paths = sorted(list(mask_in_dir.glob("*.tif")))

    # Exclude the contrast enhanced and bad segmentations
    exclude = [
        read.is_excluded(
            read.fish_number(f), exclude_train_data=False, exclude_unknown_age=False
        )
        for f in img_paths
    ]

    mask_paths = [m for m, e in zip(mask_paths, exclude) if not e]
    img_paths = [i for i, e in zip(img_paths, exclude) if not e]
    # Read in the masks

    masks = [tifffile.imread(f) for f in tqdm(mask_paths)]
    # Read in the greyscale
    imgs = [tifffile.imread(f) for f in tqdm(img_paths)]
    # Get the metadata

    metadata = [read.metadata(read.fish_number(f)) for f in img_paths]

    params_file = "radiomics_config.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

    cases = [
        (img_array, mask_array, m)
        for (img_array, mask_array, m) in zip(imgs, masks, metadata)
    ]

    features_list = []
    for img_array, mask_array, mdata in tqdm(cases):
        # Convert numpy arrays to SimpleITK images
        img = sitk.GetImageFromArray(img_array)
        mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))

        img.SetSpacing(mdata.voxel_size)
        mask.SetSpacing(mdata.voxel_size)

        # Extract features
        result = extractor.execute(img, mask)

        # Keep only numeric features
        result_clean = {
            k: v for k, v in result.items() if isinstance(v, (int, float, np.ndarray))
        }
        result_clean["ID"] = mdata.n

        features_list.append(result_clean)
    features_df = pd.DataFrame(features_list).set_index("ID")

    features_df.to_csv("features.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    main()
