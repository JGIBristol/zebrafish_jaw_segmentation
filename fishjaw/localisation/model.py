"""
Model arch and training loop

"""

import torch
from tqdm import tqdm
from monai.networks.nets import AttentionUnet


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
