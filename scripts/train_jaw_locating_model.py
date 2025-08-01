"""
Train a model to locate the zebrafish jaw.

We'll want to do this because we want to segment the jaw out from a CT scan of the
whole fish - the first step will be to crop the jaw out from it, and then we can
use the model trained in `train_model.py` to segment the jaw.

"""

import pathlib
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from fishjaw.images import io
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


def main(model_name: str, debug_plots: bool) -> None:
    """
    Read (cached) downsampled dicoms, init a model and train it to localise the jaw.

    The jaw centre will be the centroid of the segmentation mask; we will use a heatmap
    with a gradually shrinking kernel to train the model. Then we will recover
    the jaw centre from the heatmap by convolving to find its centre.

    """
    config = util.userconf()["jaw_loc_config"]
    out_dir = files.script_out_dir() / "jaw_location"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        _cache_dicoms(
            target_size=config["downsampled_dicom_size"],
            dicom_paths=dicom_paths,
            downsampled_paths=downsampled_paths,
        )

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

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["n_workers"],
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
        fig.savefig(out_dir / "train_heatmap.png")
        plt.close(fig)

    net = model.get_model(config["model_name"])

    # Train it
    net, train_losses, val_losses = train(
        net,
        train_loader,
        val_loader,
        config["learning_rate"],
        config["num_epochs"],
        config["device"],
    )

    # Plot losses
    plot_losses(train_losses, val_losses)

    # Plot heatmaps for training + val data
    if debug_plots:
        for loader, name in zip([train_loader, val_loader], ["train", "val"]):
            img, _ = next(iter(loader))
            prediction = net(img.to(config["device"])).cpu().detach().numpy()
            fig, _ = plotting.plot_heatmap(img, prediction)
            fig.savefig(out_dir / f"{name}_heatmap_prediction.png")
            plt.close(fig)

    # Read in the original and downsampled test data
    # We may want to plot the heatmap on the downsampled data (for debug)
    # Also plot the actual/predicted centre on the original size image
    test_img, test_label = io.read_dicom(dicom_paths[-1])
    downsampled_test_img, downsampled_test_label = io.read_dicom(downsampled_paths[-1])

    # Plot heatmap
    if debug_plots:
        fig, _ = plotting.plot_heatmap(
            torch.tensor(test_img).unsqueeze(0).unsqueeze(0),
            torch.tensor(test_label).unsqueeze(0).unsqueeze(0),
        )
        fig.savefig(out_dir / "test_heatmap.png")
        plt.close(fig)

    # Find the predicted centroid
    predicted_centroid = model.predict_centroid(
        net,
        torch.tensor(downsampled_test_img)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(config["device"]),
    )
    if debug_plots:
        # Plot the centroid on the downsampled image
        ...

    # Find the scale factor
    scaled_predicted_centroid = data.scale_prediction_up(
        predicted_centroid,
        data.scale_factor(test_img.shape, downsampled_test_img.shape),
    )

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
    parser.add_argument(
        "--debug-plots",
        action="store_true",
        help="Plot the training data and downsampled testing data/heatmaps for test data."
        "Losses and upsampled point estimate on test data are always plotted",
    )

    main(**vars(parser.parse_args()))
