"""
Model arch and training loop

"""

import pathlib
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.networks.nets import AttentionUnet

from ..images.transform import crop as _crop
from .data import downsample_img, scale_prediction_up, scale_factor, HeatmapDataset
from . import plotting


@dataclass
class TrainMetrics:
    """
    Training metrics for the jaw localisation model
    """

    model: torch.nn.Module
    """ The trained model """

    train_losses: list[list[float]]
    val_losses: list[list[float]]

    train_kl: list[list[float]]
    val_kl: list[list[float]]

    train_dice: list[list[float]]
    val_dice: list[list[float]]

    train_mse: list[list[float]]
    val_mse: list[list[float]]


def get_model(device) -> AttentionUnet:
    """
    Hard-coded architecture - I don't really care about squeezing performance
    out of this model, we just need it to give us a reasonable cropping window

    """
    return AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        strides=(2, 2, 2, 2, 2),
        channels=(8, 16, 32, 64, 128),
        dropout=0.1,
    ).to(device)


def kl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence loss
    """
    B = pred.size(0)
    pred = pred.view(B, -1)
    target = target.view(B, -1)

    eps = 1e-8

    target = target / (target.sum(dim=1, keepdim=True) + eps)

    pred = torch.nn.functional.log_softmax(pred, dim=1)

    return torch.nn.functional.kl_div(pred, target, reduction="batchmean")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Distance between predicted and target heatmaps

    """
    # Apply activation fcn to pred to convert to probability distribution
    pred = torch.nn.functional.softmax(pred.view(pred.size(0), -1), dim=1).view_as(pred)

    return torch.nn.functional.mse_loss(pred, target, reduction="sum")


def dice_loss(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    """
    Dice loss for comparing predicted and target heatmaps.
    Works with continuous values (not binary masks).
    """
    B = pred.size(0)
    pred = torch.nn.functional.softmax(pred.view(B, -1), dim=1).view_as(pred)

    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)

    pred_norm = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + epsilon)
    target_norm = target_flat / (target_flat.sum(dim=1, keepdim=True) + epsilon)

    intersection = (pred_norm * target_norm).sum(dim=1)
    union = pred_norm.pow(2).sum(dim=1) + target_norm.pow(2).sum(dim=1)

    dice_score = (2.0 * intersection + epsilon) / (union + epsilon)
    loss = 1.0 - dice_score.mean()

    return loss


def _dataloader(
    dataset: torch.utils.data.Dataset, *, num_workers: int, batch_size: int, train: bool
) -> torch.utils.data.DataLoader:
    """
    Hard-coded options for the dataloader...
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
        pin_memory=True,
        persistent_workers=False,  # Since we might modify the dataset during training
    )


def _shrink_heatmaps(
    train_data: torch.utils.data.Dataset,
    val_data: torch.utils.data.Dataset,
    batch_size: int,
    epoch: int,
    num_workers: int,
    fig_out_dir: pathlib.Path,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    During training, if we are performing well, we might want to shrink the heatmap
    to get a better estimate of where the centre is
    """
    # Reduce heatmap size
    new_sigma = train_data.get_sigma() * 0.9

    train_data.set_heatmaps(new_sigma)
    val_data.set_heatmaps(new_sigma)

    # Recreate loaders
    train_loader = _dataloader(
        train_data, num_workers=num_workers, batch_size=batch_size, train=True
    )
    val_loader = _dataloader(
        val_data, num_workers=num_workers, batch_size=batch_size, train=False
    )

    # Plot a heatmap, labelling the epoch and sigma
    fig, _ = plotting.plot_heatmap(*next(iter(train_loader)))
    fig.suptitle(f"Epoch {epoch}, sigma {train_data.get_sigma()}")
    fig.savefig(
        fig_out_dir
        / f"heatmap_{epoch=}_sigma_{train_data.get_sigma():.3f}.png".replace("=", "_")
    )
    plt.close(fig)

    return train_loader, val_loader


def train(
    model: torch.nn.Module,
    train_data: HeatmapDataset,
    val_data: HeatmapDataset,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    device: str,
    shrink_heatmap: bool,
    fig_out_dir: pathlib.Path,
) -> TrainMetrics:
    """
    Training loop, with a progress bar

    :param train_loader, val_loader: images/heatmaps (normalised)
    :param device: "cuda" or "cpu"

    :return: trained model
    :return: train losses, val losses
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = _dataloader(
        train_data, num_workers=num_workers, batch_size=batch_size, train=True
    )
    val_loader = _dataloader(
        val_data, num_workers=num_workers, batch_size=batch_size, train=False
    )

    loss_fn = kl_loss

    retval = TrainMetrics(None, [], [], [], [], [], [], [], [])
    metrics = [dice_loss, kl_loss, mse_loss]
    non_loss_fns = tuple((f for f in metrics if f != loss_fn))

    # Mapping for function objects onto names that we will use
    # to assign the right things in the return value
    mapping = {dice_loss: "dice", kl_loss: "kl", mse_loss: "mse"}

    model.train()
    try:
        pbar = tqdm(range(num_epochs), desc="Training...")
        for epoch in pbar:
            # If the loss for the last epoch was < a special value
            # then we want to shrink the heatmap
            if (
                shrink_heatmap
                and (train_data.get_sigma() > 0.5)
                and (
                    (np.max(retval.train_losses[-1]) if retval.train_losses else np.inf)
                    < 1.0
                )
            ):
                train_loader, val_loader = _shrink_heatmaps(
                    train_data, val_data, batch_size, epoch, num_workers, fig_out_dir
                )

            # Add new empty lists for this epoch
            retval.train_kl.append([])
            retval.val_kl.append([])
            retval.train_dice.append([])
            retval.val_dice.append([])
            retval.train_mse.append([])
            retval.val_mse.append([])

            for image, heatmap in train_loader:
                image, heatmap = image.to(device), heatmap.to(device)

                optimiser.zero_grad()

                outputs = model(image)
                loss = loss_fn(outputs, heatmap)

                loss.backward()
                optimiser.step()

                getattr(retval, f"train_{mapping[loss_fn]}")[-1].append(loss.item())

                # Evaluate the other metrics
                for fn in non_loss_fns:
                    getattr(retval, f"train_{mapping[fn]}")[-1].append(
                        fn(outputs, heatmap).item()
                    )

            for image, heatmap in val_loader:
                image, heatmap = image.to(device), heatmap.to(device)
                with torch.no_grad():
                    outputs = model(image.to(device))
                    for fn in metrics:
                        getattr(retval, f"val_{mapping[fn]}")[-1].append(
                            fn(outputs, heatmap).item()
                        )

            # Assign the losses in the metrics object too
            retval.train_losses = getattr(retval, f"train_{mapping[loss_fn]}")
            retval.val_losses = getattr(retval, f"val_{mapping[loss_fn]}")

            pbar.set_postfix(
                train_loss=np.mean(retval.train_losses[-1]),
                val_loss=np.mean(retval.val_losses[-1]),
            )
    except KeyboardInterrupt:
        # remove any empty lists
        for l in [
            retval.train_kl,
            retval.val_kl,
            retval.train_dice,
            retval.val_dice,
            retval.train_mse,
            retval.val_mse,
            retval.train_losses,
            retval.val_losses,
        ]:
            if [] in l:
                l.remove([])
        print("Training interrupted...")

    retval.model = model

    return retval


def heatmap(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    """
    Get the heatmap prediction for a single image

    This function tries to identify which device the model is on and
    performs the inference there. This will break if the model
    is on multiple devices, but what are the chances of that?

    :param model: trained jaw localisation model
    :param image: image to predict on. Should be on the CPU

    :return: heatmap prediction as a numpy array

    """
    # NB this will break if the model is on multiple devices...
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        heatmap_ = model(
            torch.tensor(image.astype(np.float32), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

    # Apply activation
    # Use softmax instead of sigmoid since the model returns logits and we
    # want to convert them to probabilities
    B = heatmap_.size(0)
    heatmap_ = torch.nn.functional.softmax(heatmap_.view(B, -1), dim=1).view_as(
        heatmap_
    )

    return heatmap_.squeeze().cpu().numpy()


def _heatmap_center(heatmap: torch.Tensor) -> list[tuple[int, int, int]]:
    """
    Find the center of the heatmap(s) by convolving with a Gaussian

    :param heatmap: 5D tensor (batch, channel, z, y, x)
    """
    kernel_size, sigma = 5, 1.0

    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")

    kernel = torch.exp(-(z**2 + y**2 + x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, *kernel.shape)

    smoothed_heatmap = torch.nn.functional.conv3d(
        heatmap, kernel, padding=kernel_size // 2
    )

    # Find the index of the maximum value in the smoothed heatmap
    flat_idx = torch.argmax(smoothed_heatmap.view(smoothed_heatmap.size(0), -1), dim=1)

    # Convert it to a 3d coord
    batch_size, _, z_size, y_size, x_size = smoothed_heatmap.shape
    z = flat_idx // (y_size * x_size)
    y = (flat_idx % (y_size * x_size)) // x_size
    x = flat_idx % x_size

    return [(z[i].item(), y[i].item(), x[i].item()) for i in range(batch_size)]


def predict_centroid(model: torch.nn.Module, image: np.ndarray) -> tuple[int, int, int]:
    """
    Predict the centroid of the jaw from an image using the trained model

    This function tries to identify which device the model is on and
    performs the inference there. This will break if the model
    is on multiple devices, but what are the chances of that?

    :param model: trained model
    :param image: 3D np array (z, y, x) - i.e. one sample

    :return: predicted centroid as a tuple (z, y, x)
    """
    predicted_heatmap = heatmap(model, image)

    (centroid,) = _heatmap_center(
        torch.Tensor(predicted_heatmap).unsqueeze(0).unsqueeze(0)
    )

    return centroid


def crop(
    model: torch.nn.Module,
    image: np.ndarray,
    model_input_size: tuple[int, int, int],
    window_size: tuple[int, int, int],
) -> np.ndarray:
    """
    Crop around the centroid identified by the model

    :param model: trained jaw localisation model
    :param image: 3D np array (z, y, x) - i.e. one sample
    :param model_input_size: size of the images the model expects
    :param window_size: size of the crop window (z, y, x)

    :return: cropped image as a numpy array
    """
    # Find the centroid on the downsampled image
    centroid = predict_centroid(
        model, downsample_img(image, model_input_size, interpolate=True)
    )

    # Scale the centroid back up to the original image size
    centroid = scale_prediction_up(
        centroid, scale_factor(image.shape, model_input_size)
    )

    return _crop(image, centroid, window_size, centred=True)
