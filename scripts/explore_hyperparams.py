"""
Train several models with different hyperparameters to compare the results

"""

import pathlib
import argparse

import yaml
import torch
import numpy as np
import monai.losses
import torchio as tio
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Union

from fishjaw.util import files
from fishjaw.images import transform
from fishjaw.model import data, model
from fishjaw.visualisation import images_3d, training


def _output_parent(mode: str) -> pathlib.Path:
    """Parent dir for output"""
    retval = files.script_out_dir() / "tuning_output" / mode
    if not retval.is_dir():
        retval.mkdir(parents=True)
    return retval


def _output_dir(n: int, mode: str) -> pathlib.Path:
    """
    Get the output directory for this run

    """
    out_dir = _output_parent(mode) / str(n)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    return out_dir


def _n_existing_dirs(mode: str) -> int:
    """
    How many runs we've done in this mode so far

    Doesn't check the integrity of the directories, just counts them

    """
    out_dir = _output_parent(mode)

    # The directories are named by number, so we can just count them
    return sum(1 for file_ in out_dir.iterdir() if file_.is_dir())


def _lr(rng: np.random.Generator, mode: str) -> float:
    if mode == "coarse":
        lr_range = (-6, 1)
    elif mode == "med":
        # This is a bit of a guess cus it never really blew up in the coarse search
        lr_range = (-6, 1)
    elif mode == "fine":
        # From centering around a value that seems to broadly work
        lr_range = (-5, 0)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return 10 ** rng.uniform(*lr_range)


def _batch_size(rng: np.random.Generator) -> int:
    # Maximum here sort of depends on what you can fit on the GPU
    return int(rng.integers(1, 32))


def _epochs(rng: np.random.Generator, mode: str) -> int:
    if mode == "coarse":
        return 5
    if mode == "med":
        return 15
    return int(rng.integers(25, 500))


def _alpha(rng: np.random.Generator) -> float:
    return rng.uniform(0.0, 1.0)


def _n_filters(rng: np.random.Generator) -> int:
    return int(rng.integers(3, 16))


def _lambda(rng: np.random.Generator, mode: str) -> float:
    if mode != "fine":
        return 1
    return 1 - (10 ** rng.uniform(-17, -1))


def _config(rng: np.random.Generator, mode: str) -> dict:
    """
    Options for the model

    """
    alpha = _alpha(rng)

    config = {
        "model_params": {
            "model_name": "monai.networks.nets.AttentionUnet",
            "spatial_dims": 3,
            "n_classes": 2,
            "in_channels": 1,
            "n_layers": 6,
            "n_initial_filters": _n_filters(rng),
            "kernel_size": 3,
            "stride": 2,
            "dropout": 0.0,
        },
        "device": "cuda",
        "learning_rate": _lr(rng, mode),
        "optimiser": "Adam",
        "batch_size": _batch_size(rng),
        "epochs": _epochs(rng, mode),
        "lr_lambda": _lambda(rng, mode),
        "loss": "monai.losses.TverskyLoss",
        "loss_options": {
            "include_background": False,
            "to_onehot_y": True,
            "alpha": alpha,
            "beta": 1 - alpha,
            "sigmoid": True,
        },
        "torch_seed": 0,
        "mode": mode,
        "patch_size": "160,160,160",
        "window_size": "192,192,192",
        "dicom_dir": "/home/mh19137/zebrafish_jaw_segmentation/dicoms/",
        "validation_dicoms": ["ak_39", "ak_86"],
        "test_dicoms": ["ak_131"],
        "transforms": {
            "torchio.RandomFlip": {"axes": [0, 1, 2], "flip_probability": 0.5},
            "torchio.RandomAffine": {"p": 0.25, "degrees": 10, "scales": 0.2},
        },
    }

    return config


def train_model(
    config: dict,
    data_config: data.DataConfig,
) -> tuple[torch.nn.Module, list[list[float]], list[list[float]]]:
    """
    Create a model, train and return it

    Returns the model, the training losses and the validation losses

    """
    # Create a model and optimiser
    net = model.model(config["model_params"])
    device = config["device"]
    net = net.to(device)

    # Get the loss options
    _, loss_name = config["loss"].rsplit(".", 1)
    loss_options = config["loss_options"]

    optim = model.optimiser(config, net)
    train_config = model.TrainingConfig(
        device,
        config["epochs"],
        torch.optim.lr_scheduler.ExponentialLR(optim, gamma=config["lr_lambda"]),
    )
    return model.train(
        net,
        optim,
        getattr(monai.losses, loss_name)(**loss_options),
        data_config,
        train_config,
    )


def step(
    config: dict,
    data_config: data.DataConfig,
    out_dir: pathlib.Path,
    full_validation_subjects: Union[list[tio.Subject], None],
):
    """
    Get the right data, train the model and create some outputs

    """
    # Set the seed before training to hopefully make things a bit more deterministic
    # (nb torch isn't fully deterministic anyway)
    torch.manual_seed(config["torch_seed"])
    net, train_losses, val_losses = train_model(config, data_config)

    if config["mode"] != "coarse":
        # Plot a training patch
        patch = next(iter(data_config.train_data))
        fig, _ = images_3d.plot_slices(
            patch[tio.IMAGE][tio.DATA].squeeze()[0].numpy(),
            patch[tio.LABEL][tio.DATA].squeeze()[0].numpy(),
        )
        fig.savefig(str(out_dir / "train_patch.png"))
        plt.close(fig)

        # Plot the loss
        fig = training.plot_losses(train_losses, val_losses)
        fig.suptitle(f"Trained for {len(train_losses)} epochs of {config['epochs']}")
        fig.tight_layout()
        fig.savefig(str(out_dir / "loss.png"))
        plt.close(fig)

        # We need to find the activation for inference
        activation = model.activation_name(config)

        # Plot the validation data
        one_patch = next(iter(data_config.val_data))
        val_subject = tio.Subject(
            image=tio.Image(
                tensor=one_patch[tio.IMAGE][tio.DATA][0], type=tio.INTENSITY
            ),
            label=tio.Image(tensor=one_patch[tio.LABEL][tio.DATA][0], type=tio.LABEL),
        )
        fig = images_3d.plot_inference(
            net,
            val_subject,
            patch_size=data.get_patch_size(config),
            patch_overlap=(4, 4, 4),
            activation=activation,
        )
        fig.savefig(str(out_dir / "val_pred.png"))
        plt.close(fig)

        # Plot the ground truth for this image
        fig, _ = images_3d.plot_subject(val_subject)
        fig.savefig(str(out_dir / "val_truth.png"))
        plt.close(fig)

        # Save the ground truth and validation prediction to file
        np.save(out_dir / "val_truth.npy", val_subject[tio.LABEL][tio.DATA].numpy())

        prediction = model.predict(
            net,
            val_subject,
            patch_size=data.get_patch_size(config),
            patch_overlap=(4, 4, 4),
            activation=activation,
        )
        np.save(out_dir / "val_pred.npy", prediction)

    # Save the losses to file
    np.save(out_dir / "train_losses.npy", train_losses)
    np.save(out_dir / "val_losses.npy", val_losses)


def main(*, mode: str, n_steps: int, continue_run: bool):
    """
    Set up the configuration and run the training

    """
    # Check if we have existing directories
    n_existing_dirs = _n_existing_dirs(mode)
    if continue_run:
        if n_existing_dirs == 0:
            raise ValueError("No existing directories to continue from")
        start = n_existing_dirs
    else:
        if n_existing_dirs > 0:
            raise ValueError("Existing directories found")
        start = 0

    rng = np.random.default_rng()

    # Get the data we need for training
    # We don't want to apply any transforms if we're not doing the fine search
    example_config = _config(rng, mode)
    if mode != "fine":
        example_config["transforms"] = {}
    # Throw away the test data - we don't need it for hyperparam tuning (that would be cheating)
    train_subjects, val_subjects, _ = data.read_dicoms_from_disk(example_config)

    # Additionally, we'll want to get tensors representing the (full, not patch-wise) validation
    # data so that we can make plots from it - we don't want our validation Dice score to
    # depend on the patch that was chosen
    full_validation_subjects = (
        [
            data.subject(path, transform.window_size(example_config))
            for path in tqdm(
                files.dicom_paths(example_config, "val"),
                desc=f"Reading {mode} DICOMs, again",
            )
        ]
        if mode != "coarse"  # We don't need this for the quick search
        else None
    )

    for i in range(start, start + n_steps):
        out_dir = _output_dir(i, mode)
        config = _config(rng, mode)

        # Since the dataloader picks random patches, the training data is slightly different
        # between runs. Hopefully this doesn't matter though
        data_config = data.DataConfig(config, train_subjects, val_subjects)

        with open(out_dir / "config.yaml", "w", encoding="utf-8") as cfg_file:
            yaml.dump(config, cfg_file)

        try:
            step(config, data_config, out_dir, full_validation_subjects)
        except torch.cuda.OutOfMemoryError as e:
            print(config)
            print(e)


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
    parser.add_argument(
        "--continue_run",
        action="store_true",
        help="If some directories with outputs exist, continue from where they left off",
    )

    main(**vars(parser.parse_args()))
