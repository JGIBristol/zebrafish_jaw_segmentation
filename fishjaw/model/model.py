"""
Define the model

"""

import os
import importlib
from math import sqrt

import torch
import numpy as np
import torchio as tio
from tqdm import trange
from monai.networks.nets import AttentionUnet

from ..util import util


def convert_params(params: dict) -> dict:
    """
    Convert the parameters to the right format, adding any extras
    and renaming them to be consistent with the monai API

    """
    # Get the number of channels for each layer by finding the number channels in the first layer
    # and then doing some maths
    start = int(sqrt(params["n_initial_filters"]))
    channels_per_layer = [2**n for n in range(start, start + params["n_layers"])]
    params["channels"] = channels_per_layer

    # Convolution stride is always the same, apart from in the first layer where it's implicitly 1
    # (to preserve the size of the input)
    strides = [params["stride"]] * (params["n_layers"] - 1)
    params["strides"] = strides

    # Rename some of the parameters to be consistent with the monai API
    params["out_channels"] = params.pop("n_classes")

    # Remove unused parameters
    params.pop("n_initial_filters")
    params.pop("n_layers")
    params.pop("stride")

    return params


def model_params() -> dict:
    """
    Get the model params from the user config file

    """
    return convert_params(util.userconf()["model_params"])


def monai_unet(*, params: dict = model_params()) -> AttentionUnet:
    """
    U-Net model for segmentation

    """
    return AttentionUnet(**params)


def optimiser(model: AttentionUnet) -> torch.optim.Optimizer:
    """
    Get the right optimiser by reading the user config file

    :param model: the model to optimise
    :returns: the optimiser

    """
    user_config = util.userconf()
    return getattr(torch.optim, user_config["optimiser"])(
        model.parameters(), user_config["learning_rate"]
    )


def _get_data(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the image and labels from each entry in a batch

    """
    # Each batch contains an image, a label and a location (which we don't care about)
    # We also just want to use the data (tio.DATA) from each of these
    x = data[tio.IMAGE][tio.DATA]
    y = data[tio.LABEL][tio.DATA]

    return x, y


def lossfn() -> torch.nn.modules.Module:
    """
    Get the loss function from the config file

    """
    module_path, class_name = util.userconf()["loss"].rsplit(".", 1)
    options: dict = util.userconf()["loss_options"]

    module = importlib.import_module(module_path)

    return getattr(module, class_name)(**options)


def train_step(
    model: AttentionUnet,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> tuple[AttentionUnet, list[float]]:
    """
    Train the model for one epoch, on the given batches of data provided as a dataloader

    :param model: the model to train
    :param optim: the optimiser to use
    :param loss_fn: the loss function to use
    :param train_data: the training data
    :param device: the device to run the model on

    :returns: the trained model
    :returns: list of training batch losses

    """
    # If the gradient is too large, we might want to clip it
    # Setting this to some reasonable value might help
    # (find with this, put after loss.backward():)
    # total_norm = 0
    # for p in model.parameters():
    #    if p.grad is not None:
    #        param_norm = p.grad.data.norm(2)
    #        total_norm += param_norm.item() ** 2
    # if total_norm ** 0.5 > max_grad:
    #     print(total_norm ** 0.5)
    max_grad = np.inf

    model.train()

    train_losses = []
    for data in train_data:
        x, y = _get_data(data)

        input_, target = x.to(device), y.to(device)

        optim.zero_grad()
        out = model(input_)

        loss = loss_fn(out, target)
        train_losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        optim.step()

    return model, train_losses


def validation_step(
    model: AttentionUnet,
    loss_fn: torch.nn.Module,
    validation_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> tuple[AttentionUnet, list[float]]:
    """
    Find the loss on the validation data

    :param model: the model to train
    :param loss_fn: the loss function to use
    :param train_data: the validation data
    :param device: the device to run the model on

    :returns: the trained model
    :returns: validation loss for each batch

    """
    model.eval()

    losses = np.ones(len(validation_data)) * np.nan

    for i, data in enumerate(validation_data):
        x, y = _get_data(data)

        batch_img, batch_label = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(batch_img)
            loss = loss_fn(out, batch_label)
            losses[i] = loss.item()

    return model, losses


def _save_checkpoint(
    model: AttentionUnet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: str = "checkpoints",
):
    """
    Save the model and optimizer state dictionaries to a checkpoint file.

    :param model: The model to save.
    :param optimizer: The optimizer to save.
    :param epoch: The current epoch number.
    :param checkpoint_dir: The directory to save the checkpoint file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def train(
    model: AttentionUnet,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    checkpoint: bool = False,
) -> tuple[AttentionUnet, list[list[float], list[list[float]]]]:
    """
    Train the model for the given number of epochs

    :param model: the model to train
    :param optimiser: the optimiser to use
    :param loss_fn: the loss function to use
    :param train_data: the training data
    :param validation_data: the validation data
    :param device: the device to run the model on
    :param epochs: the number of epochs to train for
    :param lr_scheduler: optional learning rate scheduler to use
    :param checkpoint: whether to checkpoint the model after each epoch

    :returns: the trained model
    :returns: list of training batch losses
    :returns: list of validation batch losses

    """
    train_batch_losses = []
    val_batch_losses = []

    progress_bar = trange(epochs, desc="Training")
    for epoch in progress_bar:
        model, train_batch_loss = train_step(
            model, optim, loss_fn, train_data, device=device
        )
        train_batch_losses.append(train_batch_loss)

        model, val_batch_loss = validation_step(
            model, loss_fn, validation_data, device=device
        )
        val_batch_losses.append(val_batch_loss)

        # Checkpoint the model
        checkpoint_interval = 5
        if checkpoint and not (epoch) % checkpoint_interval:
            _save_checkpoint(model, optim, epoch)

        # We might want to adjust the learning rate during training
        if lr_scheduler:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(np.mean(val_batch_losses[-1]))
            else:
                lr_scheduler.step()
        progress_bar.set_description(f"Val loss: {np.mean(val_batch_losses[-1]):.4f}")

    return model, train_batch_losses, val_batch_losses


def _predict_patches(
    model: torch.nn.Module,
    patches: tio.data.sampler.PatchSampler,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Make a prediction some patches

    Returns the predictions and the locations of the patches

    """
    tensors = []
    locations = []
    for patch in patches:
        tensors.append(patch[tio.IMAGE][tio.DATA].unsqueeze(0))
        locations.append(patch[tio.LOCATION])

    device = next(model.parameters()).device

    tensors = torch.cat(tensors, dim=0).to(device)
    locations = torch.stack(locations)

    return model(tensors).to("cpu").detach(), locations


def predict(
    model: torch.nn.Module,
    subject: tio.Subject,
    *,
    patch_size: tuple[int, int, int],
    patch_overlap: tuple[int, int, int],
    activation: str
) -> np.ndarray:
    """
    Make a prediction on a subject using the provided model

    :param model: the model to use
    :param subject: the subject to predict on
    :param patch_size: the size of the patches to use
    :param patch_overlap: the overlap between patches. Uses a hann window
    :param activation: the activation function to use

    """
    assert activation in {"softmax", "sigmoid"}

    # Make predictions on the patches
    sampler = tio.GridSampler(subject, patch_size, patch_overlap=patch_overlap)
    prediction, locations = _predict_patches(model, sampler)

    # Apply the activation function
    if activation == "softmax":
        prediction = torch.nn.functional.softmax(prediction, dim=1)
    elif activation == "sigmoid":
        prediction = torch.sigmoid(prediction)
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    # Stitch them together
    aggregator = tio.inference.GridAggregator(sampler=sampler, overlap_mode="hann")
    aggregator.add_batch(prediction, locations=locations)

    return aggregator.get_output_tensor()[1].numpy()
