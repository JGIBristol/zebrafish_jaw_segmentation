"""
Train a model to locate the zebrafish jaw.

We'll want to do this because we want to segment the jaw out from a CT scan of the
whole fish - the first step will be to crop the jaw out from it, and then we can
use the model trained in `train_model.py` to segment the jaw.

"""

import pathlib
import argparse

from tqdm import tqdm

from fishjaw.images import io
from fishjaw.util import util
from fishjaw.localisation import data


def main(model_name: str):
    """
    Read (cached) downsampled dicoms, init a model and train it to localise the jaw.

    The jaw centre will be the centroid of the segmentation mask; we will use a heatmap
    with a gradually shrinking kernel to train the model. Then we will recover
    the jaw centre from the heatmap by convolving to find its centre.

    """
    config = util.userconf()

    # TODO config option
    input_dirs = [
        pathlib.Path("dicoms") / "Training set 2",
        pathlib.Path("dicoms") / "Training set 4 (Wahab resegmented by felix)",
    ]
    dicom_paths = sorted(
        [path for input_dir in input_dirs for path in input_dir.glob("**/*.dcm")]
    )

    downsampled_paths = [data.downsampled_dicom_path(p) for p in dicom_paths]
    parent_dirs = set(p.parent for p in downsampled_paths)
    assert len(parent_dirs) == len(
        input_dirs
    ), "Should have the same number of downsampled dicom dirs as input dicom dirs"
    for parent_dir in parent_dirs:
        parent_dir.mkdir(parents=True, exist_ok=True)

    if not all(p.exists() for p in downsampled_paths):
        target_shape = config["downsampled_dicom_size"]
        pbar = tqdm(zip(dicom_paths, downsampled_paths), total=len(dicom_paths))

        for in_path, out_path in zip(dicom_paths, downsampled_paths):
            if not out_path.exists():
                pbar.set_description(f"Reading {in_path.name}")
                img, label = io.read_dicom(in_path)

                pbar.set_description(f"Downsampling {in_path.name}")
                img, label = data.downsample(img, label, target_shape)

                # Create a dicom and save it
                dicom = data.write_dicom(img, label, out_path)
            pbar.update(1)

    # Read in the downsampled dicoms
    # Leave the last one for testing
    train_imgs, train_labels = zip(*[io.read_dicom(p) for p in downsampled_paths[:-4]])
    val_imgs, val_labels = zip(*[io.read_dicom(p) for p in downsampled_paths[-4:-1]])

    # Set up training data heatmaps
    train_data = data.HeatmapDataset(
        images=train_imgs,
        masks=train_labels,
        sigma=config["initial_kernel_size"],
    )
    val_data = data.HeatmapDataset(
        images=val_imgs,
        masks=val_labels,
        sigma=config["initial_kernel_size"],
    )

    # Plot the first training data heatmap

    # Set up the model
    # Train it
    # Plot losses
    # Plot heatmaps for training + val data
    # Apply the model to the downsampled test data
    # Plot heatmap
    # Read in the original test data
    # Find the scale factor
    # Scale the prediction back up to original size
    # Plot the predicted centroid on the original image
    # Crop using the prediction, save the image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model to locate the zebrafish jaw."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="locator",
    )

    main(**vars(parser.parse_args()))
