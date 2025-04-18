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

from fishjaw.util import files
from fishjaw.images import transform
from fishjaw.model import data, model
from fishjaw.visualisation import images_3d, training


def _output_parent(mode: str, out_dir: pathlib.Path) -> pathlib.Path:
    """Parent dir for output"""
    retval = files.script_out_dir() / out_dir / mode
    if not retval.is_dir():
        retval.mkdir(parents=True)
    return retval


def _output_dir(n: int, mode: str, out_dir: pathlib.Path) -> pathlib.Path:
    """
    Get the output directory for this run

    """
    out_dir = _output_parent(mode, out_dir) / str(n)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    return out_dir


def _start_dir(mode: str, out_dir: pathlib.Path, continue_run: bool) -> int:
    """
    Which directory we should start from, based on whether we have some already and
    if we've been given the flag to continue

    Doesn't check the integrity of the directories, just counts them

    """
    # The directories are named by number, so we can just count them
    n_existing_dirs = sum(
        1 for file_ in _output_parent(mode, out_dir).iterdir() if file_.is_dir()
    )
    if continue_run:
        if n_existing_dirs == 0:
            raise ValueError("No existing directories to continue from")
        return n_existing_dirs
    if n_existing_dirs > 0:
        raise ValueError("Existing directories found")
    return 0


def _lr(rng: np.random.Generator, mode: str) -> float:
    if mode == "coarse":
        lr_range = (-6, 1)
    elif mode == "med":
        # This is a bit of a guess cus it never really blew up in the coarse search
        lr_range = (-6, 1)
    elif mode == "fine":
        # From centering around a value that seems to broadly work
        lr_range = (-3.5, -2)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return 10 ** rng.uniform(*lr_range)


def _batch_size(rng: np.random.Generator, mode: str) -> int:
    # Maximum here sort of depends on what you can fit on the GPU
    if mode == "fine":
        return int(rng.integers(1, 21))
    return int(rng.integers(1, 33))


def _epochs(mode: str) -> int:
    if mode == "coarse":
        return 5
    if mode == "med":
        return 15
    # return int(rng.integers(100, 500))
    return 400


def _alpha(rng: np.random.Generator) -> float:
    return 10 ** rng.uniform(-4, 0)


def _n_filters(rng: np.random.Generator) -> int:
    return int(rng.integers(3, 16))


def _lambda(rng: np.random.Generator, mode: str) -> float:
    if mode != "fine":
        return 1
    return 1 - (10 ** rng.uniform(-12, -1))


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
        "batch_size": _batch_size(rng, mode),
        "epochs": _epochs(mode),
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
        "dicom_dirs": [
            "/home/mh19137/zebrafish_jaw_segmentation/dicoms/Training set 1/",
            "/home/mh19137/zebrafish_jaw_segmentation/dicoms/Training set 2/",
            "/home/mh19137/zebrafish_jaw_segmentation/dicoms/Training set 3 (base of jaw)/",
        ],
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
    full_validation_subjects: list[tio.Subject],
    batch_size: int = 1,
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
            # Squeeze out the channel dim
            # don't want to accidentally squeeze out the batch dim
            # if batch size is 1, so pass it explicitly
            patch[tio.IMAGE][tio.DATA].squeeze(dim=1)[0].numpy(),
            patch[tio.LABEL][tio.DATA].squeeze(dim=1)[0].numpy(),
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
        predictions = []
        for i, val_subject in enumerate(full_validation_subjects):
            fig = images_3d.plot_inference(
                net,
                val_subject,
                patch_size=data.get_patch_size(config),
                patch_overlap=(4, 4, 4),
                activation=activation,
                batch_size=batch_size,
            )
            fig.savefig(str(out_dir / f"val_pred_{i}.png"))
            plt.close(fig)

            # Plot the ground truth for this image
            fig, _ = images_3d.plot_subject(val_subject)
            fig.savefig(str(out_dir / f"val_truth_{i}.png"))
            plt.close(fig)

            # Save the ground truth and validation prediction to file
            np.save(
                out_dir / f"val_truth_{i}.npy",
                val_subject[tio.LABEL][tio.DATA]
                .squeeze(dim=0)
                .numpy(),  # Squeeze out the channel dim
            )

            prediction = model.predict(
                net,
                val_subject,
                patch_size=data.get_patch_size(config),
                patch_overlap=(4, 4, 4),
                activation=activation,
                batch_size=batch_size,
            )
            np.save(out_dir / f"val_pred_{i}.npy", prediction)
            predictions.append(prediction)

    # Save the losses to file
    np.save(out_dir / "train_losses.npy", train_losses)
    np.save(out_dir / "val_losses.npy", val_losses)


def main(*, mode: str, n_steps: int, continue_run: bool, out_dir: str) -> None:
    """
    Set up the configuration and run the training

    """
    out_dir = pathlib.Path(out_dir)

    # Check if we have existing directories
    start = _start_dir(mode, out_dir, continue_run)

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
                desc="Reading val DICOMs, again",
            )
        ]
        if mode != "coarse"  # We don't need this for the quick search
        else None
    )

    for i in range(start, start + n_steps):
        run_dir = _output_dir(i, mode, out_dir)
        config = _config(rng, mode)

        with open(run_dir / "config.yaml", "w", encoding="utf-8") as cfg_file:
            yaml.dump(config, cfg_file)

        try:
            # Since the dataloader picks random patches, the training data is slightly different
            # between runs. Hopefully this doesn't matter though
            step(
                config,
                data.DataConfig(config, train_subjects, val_subjects),
                run_dir,
                full_validation_subjects,
            )
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
        "out_dir",
        type=str,
        help="Directory to save outputs in, relative to the script output directory",
    )
    parser.add_argument(
        "--continue_run",
        action="store_true",
        help="If some directories with outputs exist, continue from where they left off",
    )

    main(**vars(parser.parse_args()))
