"""
Model arch and training loop

"""

import torch
import numpy as np
from tqdm import tqdm
from monai.networks.nets import AttentionUnet

from ..images.transform import crop as _crop
from .data import downsample_img, scale_prediction_up, scale_factor


def get_model(device) -> AttentionUnet:
    """
    Hard-coded architecture - I don't really care about squeezing performance
    out of this model, we just need it to give us a reasonable cropping window

    """
    return AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        strides=(2, 2, 2),
        channels=(4, 8, 16, 32),
        dropout=0.05,
    ).to(device)


def kl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence loss
    """
    # Apply log-softmax to predictions
    pred = torch.nn.functional.log_softmax(pred.view(pred.size(0), -1), dim=1)

    # Ensure target is normalized (if not already)
    # TODO remove this norm
    target = target.view(pred.size(0), -1)
    target = target / target.sum(dim=1, keepdim=True)

    # Compute KL divergence
    return torch.nn.functional.kl_div(pred, target, reduction="batchmean")


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    learning_rate: float,
    num_epochs: int,
    device: str,
) -> tuple[torch.nn.Module, list[list[float]], list[list[float]]]:
    """
    Training loop, with a progress bar

    :param train_loader, val_loader: images/heatmaps (normalised)
    :param device: "cuda" or "cpu"

    :return: trained model
    :return: train losses, val losses
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    # List of lists - one for each epoch
    # Each element is a list of batch losses
    train_losses, val_losses = [], []

    pbar = tqdm(range(num_epochs), desc="Training...")
    for _ in pbar:
        train_loss, val_loss = [], []

        for image, heatmap in train_loader:
            image, heatmap = image.to(device), heatmap.to(device)

            optimiser.zero_grad()

            outputs = model(image)
            loss = kl_loss(outputs, heatmap)

            loss.backward()
            optimiser.step()

            train_loss.append(loss.item())

        for image, heatmap in val_loader:
            image, heatmap = image.to(device), heatmap.to(device)
            with torch.no_grad():
                outputs = model(image.to(device))
                loss = kl_loss(outputs, heatmap.to(device))
                val_loss.append(loss.item())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        pbar.set_postfix(train_loss=np.mean(train_loss), val_loss=np.mean(val_loss))

    return model, train_losses, val_losses


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
        return (
            (
                model(
                    torch.tensor(image.astype(np.float32), dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                .cpu()
                .detach()
            )
            .squeeze()
            .numpy()
        )


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
