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


def channels(n_layers: int, n_initial_filters: int) -> list[int]:
    """
    Find the number of channels in each layer of the network

    :param n_layers: the number of layers in the network
    :param n_initial_filters: the number of filters in the first layer
    :returns: the number of filters in each layer

    """
    start = int(sqrt(n_initial_filters))
    return [2**n for n in range(start, start + n_layers)]


def optimiser(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Get the right optimiser by reading the user config file

    :param config: the configuration for the optimiser; must contain the optimiser name and learning rate
    :param model: the model to optimise
    :returns: the optimiser

    """
    return getattr(torch.optim, config["optimiser"])(
        model.parameters(), config["learning_rate"]
    )


def _load_class(name: str) -> type:
    """
    Load a class from a string.

    :param name: the name of the class to load. Should be in the format module.class,
                 where module can also contain "."s
    :returns: the class object

    """
    module_path, class_name = name.rsplit(".", 1)

    module = importlib.import_module(module_path)

    return getattr(module, class_name)


def lossfn(config: dict) -> torch.nn.modules.Module:
    """
    Get the loss function from the config file

    """
    return _load_class(config["loss"])(**config["loss_options"])


def model_params(in_params: dict) -> dict:
    """
    Find the parameters that we need to pass to the model constructor

    Converts some to the right format, adds any extras, and renames them to be
    consistent with the monai API

    :param config: configuration, that might be e.g. the
                   "model_params" dict in userconf.yml
    :returns: the parameters to pass to the model constructor

    """
    # Some parameters we can just directly take from the config
    out_params = {
        "spatial_dims": in_params["spatial_dims"],
        "in_channels": in_params["in_channels"],
        "kernel_size": in_params["kernel_size"],
        "up_kernel_size": in_params["kernel_size"],
        "dropout": in_params["dropout"],
    }

    # Others we need to calculate

    # Get the number of channels for each layer by finding the number channels in the first layer
    # and then doing some maths
    out_params["channels"] = channels(
        in_params["n_layers"], in_params["n_initial_filters"]
    )

    # Convolution stride is always the same, apart from in the first layer where it's implicitly 1
    # (to preserve the size of the input)
    out_params["strides"] = [in_params["stride"]] * (in_params["n_layers"] - 1)

    # Rename some of the parameters to be consistent with the monai API
    out_params["out_channels"] = in_params["n_classes"]

    return out_params


def model(config: dict) -> torch.nn.Module:
    """
    U-Net model for segmentation

    :param params: the configuration needed, as might be read from the model_params dict in userconf.yml
                   Must contain the following keys:
                     - model_name: the name of the model to use
                     - all the params needed for the model
    :returns: the model

    """
    # Find which model to use
    classname = _load_class(config["model_name"])

    # Parse the parameters from the config
    return classname(**model_params(config))


def _get_data(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the image and labels from each entry in a batch

    """
    # Each batch contains an image, a label and a location (which we don't care about)
    # We also just want to use the data (tio.DATA) from each of these
    x = data[tio.IMAGE][tio.DATA]
    y = data[tio.LABEL][tio.DATA]

    return x, y


def train_step(
    net: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, list[float]]:
    """
    Train the model for one epoch, on the given batches of data provided as a dataloader

    :param net: the model to train
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

    net.train()

    train_losses = []
    for data in train_data:
        x, y = _get_data(data)

        input_, target = x.to(device), y.to(device)

        optim.zero_grad()
        out = net(input_)

        loss = loss_fn(out, target)
        train_losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad)
        optim.step()

    return net, train_losses


def validation_step(
    net: torch.nn.Module,
    loss_fn: torch.nn.Module,
    validation_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, list[float]]:
    """
    Find the loss on the validation data

    :param net: the model that is being trained
    :param loss_fn: the loss function to use
    :param train_data: the validation data
    :param device: the device to run the model on

    :returns: the trained model
    :returns: validation loss for each batch

    """
    net.eval()

    losses = np.ones(len(validation_data)) * np.nan

    for i, data in enumerate(validation_data):
        x, y = _get_data(data)

        batch_img, batch_label = x.to(device), y.to(device)
        with torch.no_grad():
            out = net(batch_img)
            loss = loss_fn(out, batch_label)
            losses[i] = loss.item()

    return net, losses


def _save_checkpoint(
    model: torch.nn.Module,
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
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    checkpoint: bool = False,
) -> tuple[torch.nn.Module, list[list[float], list[list[float]]]]:
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
    net: torch.nn.Module,
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

    device = next(net.parameters()).device

    tensors = torch.cat(tensors, dim=0).to(device)
    locations = torch.stack(locations)

    return net(tensors).to("cpu").detach(), locations


def predict(
    model: torch.nn.Module,
    subject: tio.Subject,
    *,
    patch_size: tuple[int, int, int],
    patch_overlap: tuple[int, int, int],
    activation: str,
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
