"""
Train a model to segment the jawbone from labelled DICOM images

"""

import pathlib
import warnings
import argparse

import torch
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt

from fishjaw.util import util
from fishjaw.model import data, model
from fishjaw.visualisation import images_3d
from fishjaw.images import io


def plot_losses(train_losses: list[float], val_losses: list[float]) -> plt.Figure:
    """
    Plot the training and validation losses against epoch

    """
    assert len(train_losses) == len(val_losses)

    epochs = np.arange(len(train_losses))

    train_loss = np.array([np.mean(epoch_loss) for epoch_loss in train_losses])
    val_loss = np.array([np.mean(epoch_loss) for epoch_loss in val_losses])

    min_loss = min(np.min(train_loss), np.min(val_loss))
    log_train_loss = np.log(train_loss - min_loss + 1)
    log_val_loss = np.log(val_loss - min_loss + 1)

    fig, (axis, log_axis) = plt.subplots(1, 2)

    axis.plot(epochs, train_loss, label="Train")
    log_axis.plot(epochs, log_train_loss, label="Train")

    # Find quartiles - the mean might be outside this, which would be interesting wouldn't it
    train_loss_upper = [np.percentile(epoch_loss, 75) for epoch_loss in train_losses]
    train_loss_lower = [np.percentile(epoch_loss, 25) for epoch_loss in train_losses]
    axis.fill_between(epochs, train_loss_lower, train_loss_upper, alpha=0.5, color="C0")

    axis.plot(epochs, val_loss, label="Validation")
    log_axis.plot(epochs, log_val_loss, label="Train")

    val_loss_upper = [np.percentile(epoch_loss, 75) for epoch_loss in val_losses]
    val_loss_lower = [np.percentile(epoch_loss, 25) for epoch_loss in val_losses]
    axis.fill_between(epochs, val_loss_lower, val_loss_upper, alpha=0.5, color="C1")

    axis.set_title("Loss")
    axis.set_xlabel("Epoch")
    axis.legend()

    log_axis.set_title("Log Loss")
    log_axis.set_xlabel("Epoch")

    return fig


def plot_inference(
    net: torch.nn.Module,
    subject: tio.GridSampler,
    *,
    patch_size: tuple[int, int, int],
    patch_overlap: tuple[int, int, int],
) -> plt.Figure:
    """
    Plot the inference on an image

    """
    # Get the image from the subject
    image = subject[tio.IMAGE][tio.DATA].squeeze().numpy()
    print(image.shape)

    # Perform inference
    prediction = model.predict(
        net, subject, patch_size=patch_size, patch_overlap=patch_overlap
    )
    print(prediction.shape)

    fig, _ = images_3d.plot_slices(image, prediction)
    fig.suptitle("Model Prediction")
    fig.tight_layout()

    return fig


def train_model(
    config: dict,
    train_subjects: torch.utils.data.DataLoader,
    val_subjects: torch.utils.data.DataLoader,
) -> tuple[torch.nn.Module, list[list[float]], list[list[float]]]:
    """
    Create a model, train and return it

    Returns the model, the training losses and the validation losses

    """
    # Create a model and optimiser
    net = model.monai_unet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        # Yellow text
        yellow = "\033[33m"
        clear = "\033[0m"
        warnings.warn(f"{yellow}This might not be what you want!{clear}")
    print(f"Using {device} device")
    net = net.to(device)

    optimiser = model.optimiser(net)

    # Create dataloaders
    patch_size = io.patch_size()
    batch_size = config["batch_size"]
    train_loader = data.train_val_loader(
        train_subjects, train=True, patch_size=patch_size, batch_size=batch_size
    )
    val_loader = data.train_val_loader(
        val_subjects, train=False, patch_size=patch_size, batch_size=batch_size
    )

    # Define loss function
    loss = model.lossfn()

    return model.train(
        net,
        optimiser,
        loss,
        train_loader,
        val_loader,
        device=device,
        epochs=config["epochs"],
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(
            optimiser, gamma=config["lr_lambda"]
        ),
    )


def main():
    """
    Get the right data, train the model and create some outputs

    """
    config = util.userconf()
    torch.manual_seed(config["torch_seed"])
    rng = np.random.default_rng(seed=config["test_train_seed"])

    train_subjects, val_subjects, test_subject = data.get_data(rng)

    net, train_losses, val_losses = train_model(config, train_subjects, val_subjects)

    output_dir = pathlib.Path("train_output")
    if not output_dir.is_dir():
        output_dir.mkdir()

    # Plot the loss
    fig = plot_losses(train_losses, val_losses)
    fig.savefig(str(output_dir / "loss.png"))

    # Plot the testing image
    fig = plot_inference(
        net, test_subject, patch_size=io.patch_size(), patch_overlap=(4, 4, 4)
    )
    fig.savefig(str(output_dir / "prediction.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to segment the jawbone")

    main(**vars(parser.parse_args()))
