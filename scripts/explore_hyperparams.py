"""
Train several models with different hyperparameters to compare the results

"""

import pathlib
import argparse

import yaml
import torch
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt

from fishjaw.model import data, model
from fishjaw.visualisation import images_3d, training
from fishjaw.images import io


def _output_dir(n: int, mode: str) -> pathlib.Path:
    """
    Get the output directory for this run

    """
    out_dir = pathlib.Path(__file__).parents[1] / "tuning_output" / mode / str(n)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    return out_dir


def _lr(rng: np.random.Generator, mode: str) -> float:
    if mode == "coarse":
        lr_range = (-6, 1)
    elif mode == "med":
        raise NotImplementedError
    elif mode == "fine":
        raise NotImplementedError

    return 10 ** rng.uniform(*lr_range)


def _batch_size(rng: np.random.Generator, mode: str) -> int:
    # Maximum here sort of depends on what you can fit on the GPU
    return int(rng.integers(1, 12))


def _epochs(rng: np.random.Generator, mode: str) -> int:
    if mode == "coarse":
        return 3
    if mode == "med":
        return 10
    return rng.integers(50, 500)


def _alpha(rng: np.random.Generator, mode: str) -> float:
    return rng.uniform(0.0, 1.0)


def _n_filters(rng: np.random.Generator, mode: str) -> int:
    return rng.integers(4, 32)


def _config(rng: np.random.Generator, mode: str) -> dict:
    """
    Options for the model

    """
    alpha = _alpha(rng, mode)

    config = {
        "model_params": {
            "spatial_dims": 3,
            "n_classes": 2,
            "in_channels": 1,
            "n_layers": 6,
            "n_initial_filters": _n_filters(rng, mode),
            "kernel_size": 3,
            "stride": 2,
            "dropout": 0.0,
        },
        "learning_rate": _lr(rng, mode),
        "optimiser": "Adam",
        "batch_size": _batch_size(rng, mode),
        "epochs": _epochs(rng, mode),
        "lr_lambda": 1.0,
        "loss": "monai.losses.TverskyLoss",
        "loss_options": {
            "include_background": False,
            "to_onehot_y": True,
            "alpha": alpha,
            "beta": 1 - alpha,
            "softmax": True,
        },
        "torch_seed": 0,
        "mode": mode,
    }

    return config


def train_model(
    config: dict,
    train_subjects: torch.utils.data.DataLoader,
    val_subjects: torch.utils.data.DataLoader,
) -> tuple[torch.nn.Module, list[list[float]], list[list[float]]]:
    """
    Create a model, train and return it

    Returns the model, the training losses and the validation losses

    """
    # Get the model params
    model_params = model.convert_params(config["model_params"])

    # Create a model and optimiser
    net = model.monai_unet(params=model_params)
    device = "cuda"
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


def step(
    config: dict,
    train_subjects: torch.utils.data.DataLoader,
    val_subjects: torch.utils.data.DataLoader,
    test_subject: tio.Subject,
    out_dir: pathlib.Path,
):
    """
    Get the right data, train the model and create some outputs

    """
    # Set the seed before training to hopefully make things a bit more deterministic
    # (nb torch isn't fully deterministic anyway)
    torch.manual_seed(config["torch_seed"])
    net, train_losses, val_losses = train_model(config, train_subjects, val_subjects)

    if config["mode"] != "coarse":
        # Plot the loss
        fig = training.plot_losses(train_losses, val_losses)
        fig.savefig(str(out_dir / "loss.png"))
        plt.close(fig)

        # Plot the testing image
        fig = images_3d.plot_inference(
            net, test_subject, patch_size=io.patch_size(), patch_overlap=(4, 4, 4)
        )
        fig.savefig(str(out_dir / "test_pred.png"))
        plt.close(fig)

        # Plot the ground truth for this image
        fig, _ = images_3d.plot_subject(test_subject)
        fig.savefig(str(out_dir / "test_truth.png"))
        plt.close(fig)
    
    # Save the losses to file
    np.save(out_dir / "train_losses.npy", train_losses)
    np.save(out_dir / "val_losses.npy", val_losses)


def main(*, mode: str, n_steps: int):
    """
    Set up the configuration and run the training

    """
    # NB we are still using the patch_size defined in userconf
    rng = np.random.default_rng()
    train_subjects, val_subjects, test_subject = data.get_data(rng)

    for i in range(n_steps):
        out_dir = _output_dir(i, mode)

        config = _config(rng, mode)

        with open(out_dir / "config.yaml", "w") as cfg_file:
            yaml.dump(config, cfg_file)
        step(config, train_subjects, val_subjects, test_subject, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train lots of models to compare hyperparameters"
    )

    parser.add_argument(
        "mode",
        type=str,
        choices={"coarse", "med", "fine"},
        help="""Granularity of the search.
              Determines the range of hyperparameters, and which are searched""",
    )
    parser.add_argument("n_steps", type=int, help="Number of models to train")

    main(**vars(parser.parse_args()))
