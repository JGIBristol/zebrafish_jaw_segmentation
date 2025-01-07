"""
Define the model

"""

import os
import pickle
from dataclasses import dataclass
from typing import Type, Any

import torch
import numpy as np
import torchio as tio
from tqdm import trange
from torch.cuda.amp import autocast, GradScaler

from .data import DataConfig
from ..util import util, files


@dataclass(frozen=True)
class ModelState:
    """The state of the model"""

    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, torch.Tensor]

    # The configuration used to train the model, as read from the userconf.yml file
    config: dict[str, Any]

    def load_model(self, *, set_eval=True) -> torch.nn.Module:
        """
        Load the model

        Initialises the architecture from the config and loads the state dict into it
        Turns the model into evaluation mode by default

        :param eval: whether to put the model into evaluation mode

        """
        net = model(self.config["model_params"])
        net.load_state_dict(self.model_state_dict)

        if set_eval:
            net.eval()

        return net


@dataclass
class TrainingConfig:
    """The stuff needed to train a model"""

    device: torch.device
    epochs: int
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
    checkpoint: bool = False
    early_stopping: bool = False


@dataclass
class IterationConfig:
    """The stuff needed for one epoch of training"""

    net: torch.nn.Module
    optim: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_data: tio.SubjectsLoader
    scaler: GradScaler
    device: torch.device


def channels(n_layers: int, initial_channels) -> list[int]:
    """
    Get the number of channels in each layer of the network -
    starting at inital_channels and doubling every time

    :param initial_channels: the number of channels after the first convolutions
    :param n_layers: the number of layers in the network

    :returns: the number of filters in each layer of the model

    """
    return [initial_channels * 2**i for i in range(n_layers)]


def optimiser(config: dict, net: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Get the right optimiser by reading the user config file

    :param config: the configuration for the optimiser;
                   must contain the optimiser name and learning rate
    :param net: the model to optimise
    :returns: the optimiser

    """
    return getattr(torch.optim, config["optimiser"])(
        net.parameters(), config["learning_rate"]
    )


def activation_name(config: dict) -> str:
    """
    Get the name of the activation function

    I can't be bothered to do the refactor but it would be better if this returned the
    actual function wouldn't it

    :param config: the configuration, e.g. from userconf.yml
    :returns: the name of the activation function
    :raises ValueError: if no valid activation function is found in the config

    """
    if config["loss_options"].get("softmax", False):
        return "softmax"
    if config["loss_options"].get("sigmoid", False):
        return "sigmoid"
    raise ValueError("No activation found")


def lossfn(config: dict) -> torch.nn.modules.Module:
    """
    Get the loss function from the config file

    """
    loss_class: Type[torch.nn.modules.Module] = util.load_class(config["loss"])
    return loss_class(**config["loss_options"])


def model_params(in_params: dict[str, Any]) -> dict[str, Any]:
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
        in_params["n_layers"], in_params["n_initial_channels"]
    )

    # Convolution stride is always the same, apart from in the first layer where it's implicitly 1
    # (to preserve the size of the input)
    out_params["strides"] = [in_params["stride"]] * (in_params["n_layers"] - 1)

    # Rename some of the parameters to be consistent with the monai API
    out_params["out_channels"] = in_params["n_classes"]

    return out_params


def model(config: dict[str, Any]) -> torch.nn.Module:
    """
    U-Net model for segmentation

    :param params: the configuration needed, as might be read from the model_params dict in
                   userconf.yml. Must contain the following keys:
                     - model_name: the name of the model to use
                     - all the params needed for the model
    :returns: the model

    """
    # Find which model to use
    classname = util.load_class(config["model_name"])

    # Parse the parameters from the config
    return classname(**model_params(config))


def _get_data(
    data: dict[str, dict[str, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the image and labels from each entry in a batch

    """
    # Each batch contains an image, a label and a location (which we don't care about)
    # We also just want to use the data (tio.DATA) from each of these
    x = data[tio.IMAGE][tio.DATA]
    y = data[tio.LABEL][tio.DATA]

    return x, y


def _train_step(
    iteration_config: IterationConfig,
) -> tuple[torch.nn.Module, list[float]]:
    """
    Train the model for one epoch, on the given batches of data provided as a dataloader

    :param iteration_config: the stuff we need to train

    :returns: the trained model
    :returns: list of training batch losses

    """
    net = iteration_config.net
    optim = iteration_config.optim
    loss_fn = iteration_config.loss_fn
    train_data = iteration_config.train_data
    scaler = iteration_config.scaler
    device = iteration_config.device

    net.train()

    train_losses = []
    for data in train_data:
        x, y = _get_data(data)

        input_, target = x.to(device, non_blocking=True), y.to(
            device, non_blocking=True
        )

        optim.zero_grad()
        with autocast():
            out = net(input_)
            loss = loss_fn(out, target)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        train_losses.append(loss.item())

    return net, train_losses


def _validation_step(
    net: torch.nn.Module,
    loss_fn: torch.nn.Module,
    validation_data: tio.SubjectsLoader,
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


def save_checkpoint(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Save the model and optimizer state dictionaries to a checkpoint file.

    :param net: The model to save.
    :param optimizer: The optimizer to save.
    :param epoch: The current epoch number.
    :param checkpoint_dir: The directory to save the checkpoint file.
    :param config: the configuration used to train the model, as read from the userconf.yml file

    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")
    model_state = ModelState(net.state_dict(), optimizer.state_dict(), config)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(model_state, f)


def _early_stop(
    patience: int, val_losses: list[list[float]], train_losses: list[list[float]]
) -> bool:
    """
    Whether to stop training early, depending on our losses

    Will stop if:
        - the validation loss is NaN
        - the validation loss has been worse than the training loss
        - the validation loss has not decreased for `patience` epochs

    """
    assert len(val_losses) == len(train_losses)

    # We haven't trained for long enough to stop
    if len(val_losses) < patience:
        return False

    # Check if the validation loss is NaN
    mean_val_loss = np.mean(val_losses, axis=1)
    mean_train_loss = np.mean(train_losses, axis=1)
    if np.isnan(mean_val_loss[-1]):
        return True

    # Check if the validation loss has been worse than the training loss for the last few epochs
    overfit_threshhold = 1.5
    if (mean_val_loss > overfit_threshhold * mean_train_loss)[-patience:].all():
        return True

    # Check if the validation loss has not decreased for the last few epochs
    best_val_loss = np.min(mean_val_loss, axis=1)
    if (mean_val_loss[-patience:] <= best_val_loss).all():
        return True

    return False


def train(
    net: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    data_config: DataConfig,
    train_config: TrainingConfig,
) -> tuple[torch.nn.Module, list[list[float]], list[list[float]]]:
    """
    Train the model for the given number of epochs

    :param net: the model to train
    :param optimiser: the optimiser to use
    :param loss_fn: the loss function to use
    :param data_config: the data to train on
    :param train_config: the configuration for training

    :returns: the trained model
    :returns: list of training batch losses
    :returns: list of validation batch losses

    """
    train_batch_losses = []
    val_batch_losses = []

    # How many epochs to wait before stopping training
    patience = 10

    # Gradient scaler for mixed precision training
    scaler = GradScaler()

    progress_bar = trange(train_config.epochs, desc="Training")
    for _ in progress_bar:
        net, train_batch_loss = _train_step(
            IterationConfig(
                net,
                optim,
                loss_fn,
                data_config.train_data,
                scaler,
                train_config.device,
            )
        )
        train_batch_losses.append(train_batch_loss)

        net, val_batch_loss = _validation_step(
            net, loss_fn, data_config.val_data, device=train_config.device
        )
        val_batch_losses.append(val_batch_loss)

        # We might want to adjust the learning rate during training
        if train_config.lr_scheduler:
            if isinstance(
                train_config.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                train_config.lr_scheduler.step(np.mean(val_batch_losses[-1]))
            else:
                train_config.lr_scheduler.step()

        # Early stopping
        if train_config.early_stopping and _early_stop(
            patience, val_batch_losses, train_batch_losses
        ):
            break

        progress_bar.set_description(f"Val loss: {np.mean(val_batch_losses[-1]):.4f}")

    return net, train_batch_losses, val_batch_losses


def _predict_patches(
    net: torch.nn.Module,
    patches: tio.data.sampler.PatchSampler,
    batch_size: int,
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

    print(tensors.shape)

    predictions = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i : i + batch_size].to(device)
        with torch.no_grad():
            prediction = net(batch).to("cpu").detach()
        predictions.append(prediction)
        torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

    predictions = torch.cat(predictions, dim=0)

    return predictions, locations


def predict(
    net: torch.nn.Module,
    subject: tio.Subject,
    *,
    patch_size: tuple[int, int, int],
    patch_overlap: tuple[int, int, int],
    activation: str,
    batch_size: int,
) -> np.ndarray:
    """
    Make a prediction on a subject using the provided model

    :param net: the model to use
    :param subject: the subject to predict on
    :param patch_size: the size of the patches to use
    :param patch_overlap: the overlap between patches. Uses a hann window
    :param activation: the activation function to use
    :param batch_size: the batch size to use

    returns: the prediction, as a 3d numpy array

    """
    assert activation in {"softmax", "sigmoid"}

    # Make predictions on the patches
    sampler = tio.GridSampler(subject, patch_size, patch_overlap=patch_overlap)
    prediction, locations = _predict_patches(net, sampler, batch_size)

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


def load_model(model_name: str) -> ModelState:
    """
    Load a pickled model from disk given its name

    :param model_name: the name of the model to load, e.g. "model_state.pkl", as specified
                       in userconf.yml. Must end in ".pkl".

    :returns: the model

    :raises FileNotFoundError: if the model file does not exist
    :raises ValueError: if the model name does not end in ".pkl"

    """
    if not model_name.endswith(".pkl"):
        raise ValueError(f"Model name should end with .pkl: {model_name}")

    with open(files.model_path({"model_path": model_name}), "rb") as f:
        return pickle.load(f)
