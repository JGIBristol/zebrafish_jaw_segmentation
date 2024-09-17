"""
Train a model to segment the jawbone from labelled DICOM images

"""

import pickle
import pathlib
import warnings
import argparse

import torch
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt

from fishjaw.util import files, util
from fishjaw.model import data, model
from fishjaw.images import io
from fishjaw.visualisation import images_3d, training


def _plot_example(batch: dict[str, torch.Tensor]):
    """
    Plot an example of the training data

    """
    img = batch[tio.IMAGE][tio.DATA][0, 0].numpy()
    label = batch[tio.LABEL][tio.DATA][0, 0].numpy()

    fig, _ = images_3d.plot_slices(img, label)
    fig.savefig("train_output/train_example.png")


def train_model(
    config: dict,
    train_subjects: torch.utils.data.DataLoader,
    val_subjects: torch.utils.data.DataLoader,
) -> tuple[
    tuple[torch.nn.Module, list[list[float]], list[list[float]], torch.optim.Optimizer]
]:
    """
    Create a model, train and return it

    Returns the model, the training losses and the validation losses, and the optimiser

    """
    # Create a model and optimiser
    net = model.monai_unet(params=model.model_params(config["model_params"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        # Yellow text
        yellow = "\033[33m"
        clear = "\033[0m"
        warnings.warn(f"{yellow}This might not be what you want!{clear}")
    print(f"Using {device} device")
    net = net.to(device)

    optimiser = model.optimiser(config, net)

    # Create dataloaders
    patch_size = data.patch_size(config)
    batch_size = config["batch_size"]
    train_loader = data.train_val_loader(
        train_subjects, train=True, patch_size=patch_size, batch_size=batch_size
    )
    val_loader = data.train_val_loader(
        val_subjects, train=False, patch_size=patch_size, batch_size=batch_size
    )

    # Plot an example of the training data (which has been augmented)
    _plot_example(next(iter(train_loader)))

    # Define loss function
    loss = model.lossfn(config)

    return (
        model.train(
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
        ),
        optimiser,
    )


def main(*, save: bool):
    """
    Get the right data, train the model and create some outputs

    """
    config = util.userconf()
    torch.manual_seed(config["torch_seed"])
    rng = np.random.default_rng(seed=config["test_train_seed"])

    # Find the activation - we'll need this for inference
    if config["loss_options"].get("softmax", False):
        activation = "softmax"
    elif config["loss_options"].get("sigmoid", False):
        activation = "sigmoid"
    else:
        raise ValueError("No activation found")

    train_subjects, val_subjects, test_subject = data.get_data(config, rng)

    # Save the testing subject
    output_dir = pathlib.Path("train_output")
    if not output_dir.is_dir():
        output_dir.mkdir()
    with open(output_dir / "test_subject.pkl", "wb") as f:
        pickle.dump(test_subject, f)


    (net, train_losses, val_losses), optimiser = train_model(
        config, train_subjects, val_subjects
    )

    if save:
        torch.save(
            {
                "model": net.state_dict(),
                "optimiser": optimiser.state_dict(),
            },
            str(files.model_path()),
        )

    # Plot the loss
    fig = training.plot_losses(train_losses, val_losses)
    fig.savefig(str(output_dir / "loss.png"))
    plt.close(fig)

    # Plot the testing image
    fig = images_3d.plot_inference(
        net,
        test_subject,
        patch_size=data.patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=activation,
    )
    fig.savefig(str(output_dir / "test_pred.png"))
    plt.close(fig)

    # Plot the ground truth for this image
    fig, _ = images_3d.plot_subject(test_subject)
    fig.savefig(str(output_dir / "test_truth.png"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to segment the jawbone")
    parser.add_argument("--save", help="Save the model", action="store_true")
    main(**vars(parser.parse_args()))
