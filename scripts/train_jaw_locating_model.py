"""
Train a model to locate the zebrafish jaw.

We'll want to do this because we want to segment the jaw out from a CT scan of the
whole fish - the first step will be to crop the jaw out from it, and then we can
use the model trained in `train_model.py` to segment the jaw.

"""

import pathlib
import argparse

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import center_of_mass

from fishjaw.images import io
from fishjaw.images.transform import crop
from fishjaw.util import util, files
from fishjaw.localisation import data, plotting, model
from fishjaw.visualisation.training import plot_losses


def _cache_dicoms(
    target_size: tuple[int, int, int],
    dicom_paths: list[pathlib.Path],
    downsampled_paths: list[pathlib.Path],
) -> None:
    """
    Downsample and and save the dicoms to disk
    """
    pbar = tqdm(zip(dicom_paths, downsampled_paths), total=len(dicom_paths))

    for in_path, out_path in zip(dicom_paths, downsampled_paths):
        if not out_path.exists():
            pbar.set_description(f"Reading {in_path.name}")
            img, label = io.read_dicom(in_path)

            pbar.set_description(f"Downsampling {in_path.name}")
            img, label = data.downsample(img, label, target_size)

            # Create a dicom and save it
            dicom = data.write_dicom(img, label, out_path)
        pbar.update(1)


def _savefig(fig: plt.Figure, path: pathlib.Path, *, verbose: bool) -> None:
    """
    Helper function for saving figures

    Also closes the figure
    """
    if verbose:
        print(f"Saving figure to {path}")
    fig.savefig(path)
    plt.close(fig)


def _dicom_paths(config: dict) -> list[pathlib.Path]:
    """
    The paths to the training DICOMs
    """
    input_dirs = [pathlib.Path(d) for d in config["dicom_dirs"]]
    return sorted(
        [path for input_dir in input_dirs for path in input_dir.glob("**/*.dcm")]
    )


def main(model_name: str, debug_plots: bool) -> None:
    """
    Read (cached) downsampled dicoms (caching them first if required),
    init a model and train it to localise the jaw.

    The jaw centre will be the centroid of the segmentation mask; we will use a heatmap
    with a gradually shrinking kernel to train the model. Then we will recover
    the jaw centre from the heatmap by convolving to find its centre.

    """
    config = util.userconf()["jaw_loc_config"]

    # Find where the inputs are, and if necessary create the downsampled dicoms
    dicom_paths = _dicom_paths(config)
    downsampled_paths = [data.downsampled_dicom_path(p) for p in dicom_paths]
    if not all(p.exists() for p in downsampled_paths):
        _cache_dicoms(
            target_size=config["downsampled_dicom_size"],
            dicom_paths=dicom_paths,
            downsampled_paths=downsampled_paths,
        )

    parent_dirs = set(p.parent for p in downsampled_paths)
    # This checks that we haven't accidentally messed something up with the paths
    assert len(parent_dirs) == len(
        config["dicom_dirs"]
    ), "Should have the same number of downsampled dicom dirs as input dicom dirs"
    for parent_dir in parent_dirs:
        parent_dir.mkdir(parents=True, exist_ok=True)

    # Get where the outputs should go
    out_dir = files.script_out_dir() / "jaw_location" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_name}.pth"
    if model_path.exists():
        raise FileExistsError(
            f"Model already exists at {model_path}, please delete it or use a different name."
        )

    # Read in the downsampled dicoms
    # Leave the last one for testing
    train_paths = downsampled_paths[:-4]
    val_paths = downsampled_paths[-4:-1]

    test_path = dicom_paths[-1]
    downsampled_test_path = downsampled_paths[-1]

    # Set up training data heatmaps
    train_imgs, train_labels = zip(*[io.read_dicom(p) for p in train_paths])
    train_data = data.HeatmapDataset(
        images=train_imgs,
        masks=train_labels,
        sigma=config["initial_kernel_size"],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["n_workers"],
    )

    val_imgs, val_labels = zip(*[io.read_dicom(p) for p in val_paths])
    val_data = data.HeatmapDataset(
        images=val_imgs,
        masks=val_labels,
        sigma=config["initial_kernel_size"],
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["n_workers"],
    )

    if debug_plots:
        # Plot the first training data heatmap
        fig, _ = plotting.plot_heatmap(*next(iter(train_loader)))
        _savefig(fig, out_dir / "train_heatmap.png", True)

    net = model.get_model(config["device"])
    net, train_losses, val_losses = model.train(
        net,
        train_loader,
        val_loader,
        config["learning_rate"],
        config["num_epochs"],
        config["device"],
    )

    # Plot losses
    fig = plot_losses(train_losses, val_losses)
    _savefig(fig, out_dir / "losses.png", verbose=debug_plots)

    # Plot heatmaps for training + val data
    if debug_plots:
        for loader, name in zip([train_loader, val_loader], ["train", "val"]):
            img, _ = next(iter(loader))
            prediction = net(img.to(config["device"])).cpu().detach()

            fig, _ = plotting.plot_heatmap(img, prediction)
            _savefig(fig, out_dir / f"{name}_heatmap_prediction.png", verbose=True)

    with open(model_path, "wb") as f:
        torch.save(net.state_dict(), f)

    # Read in the original and downsampled test data
    # We may want to plot the heatmap on the downsampled data (for debug)
    # Also plot the actual/predicted centre on the original size image
    test_img, test_label = io.read_dicom(test_path)
    downsampled_test_img, downsampled_test_label = io.read_dicom(downsampled_test_path)

    # Plot heatmap
    predicted_heatmap = (
        net(
            torch.tensor(downsampled_test_img.astype(np.float32), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(config["device"])
        )
        .cpu()
        .detach()
    )

    if debug_plots:
        fig, _ = plotting.plot_heatmap(
            torch.tensor(test_img.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            predicted_heatmap,
        )
        _savefig(fig, out_dir / "test_heatmap.png", verbose=True)

    # Find the predicted centroid
    (predicted_centroid,) = model._heatmap_center(predicted_heatmap)
    if debug_plots:
        # Plot the centroid on the downsampled image
        fig, _ = plotting.plot_centroid(
            torch.tensor(downsampled_test_img.astype(np.float32), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
            predicted_centroid,
        )
        _savefig(fig, out_dir / "test_centroid_downsampled.png", verbose=True)

        # Plot the truth centroid
        fig, _ = plotting.plot_centroid(
            torch.tensor(test_img.astype(np.float32), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
            [int(x) for x in center_of_mass(test_label)],
        )
        _savefig(fig, out_dir / "test_centroid_truth.png", verbose=True)

    # Find the scale factor
    scaled_predicted_centroid = data.scale_prediction_up(
        predicted_centroid,
        data.scale_factor(test_img.shape, downsampled_test_img.shape),
    )

    # Plot the predicted centroid on the original image
    fig, _ = plotting.plot_centroid(
        torch.tensor(test_img.astype(np.float32)).unsqueeze(0).unsqueeze(0),
        scaled_predicted_centroid,
    )
    _savefig(fig, out_dir / "test_centroid.png", verbose=debug_plots)

    # Crop using the prediction, save the image
    cropped = crop(
        test_img,
        scaled_predicted_centroid,
        config["crop_size"],
        centred=True,
    )
    cropped_mask = crop(
        test_label,
        scaled_predicted_centroid,
        config["crop_size"],
        centred=True,
    )
    fig, _ = plotting.plot_heatmap(
        torch.tensor(cropped.astype(np.float32), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0),
        torch.tensor(cropped_mask).unsqueeze(0).unsqueeze(0),
    )
    _savefig(fig, out_dir / "test_cropped.png", verbose=debug_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model to locate the zebrafish jaw."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="locator",
    )
    parser.add_argument(
        "--debug-plots",
        action="store_true",
        help="Plot the training data and downsampled testing data/heatmaps for test data."
        "Losses and upsampled point estimate on test data are always plotted",
    )

    main(**vars(parser.parse_args()))
