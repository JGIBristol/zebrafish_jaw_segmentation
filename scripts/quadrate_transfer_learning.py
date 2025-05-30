"""
Either train a new model or fine tune an existing model on a small dataset of quadrates.

Makes some plots of the loss, inference and meshes after taking the largest connected
component of the prediction

"""

import pathlib
import argparse

import torch
import torchio as tio
import matplotlib.pyplot as plt

from fishjaw.util import util
from fishjaw.model import model
from fishjaw.transfer import data
from fishjaw.images import metrics
from fishjaw.model.data import DataConfig, get_patch_size
from fishjaw.visualisation import training, images_3d, plot_meshes
from fishjaw.inference import mesh


def train(*, epochs: int, **kwargs):
    """
    Read in the training data and use it to train a model.
    Make some plots of the loss, the inference on the testing data and output some metrics

    Saves the model
    """
    config = util.userconf()
    out_dir = (
        pathlib.Path(util.config()["script_output"])
        / "transfer_learning"
        / "quadrate"
        / "base_model"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create the right data and model setup
    train_subjects, val_subjects, test_subject = data.quadrate_data(config)

    # Hack - make the batch size the same size as the training data,
    # otherwise we'll drop all of it
    config["batch_size"] = len(train_subjects)

    quadrate_data = DataConfig(config, train_subjects, val_subjects)

    net = model.model(config["model_params"])
    net = net.to(config["device"])

    optimiser = model.optimiser(config, net)
    loss = model.lossfn(config)

    train_config = model.TrainingConfig(
        config["device"],
        epochs,
        torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=config["lr_lambda"]),
    )

    # Train and save the model
    net, train_losses, val_losses = model.train(
        net, optimiser, loss, quadrate_data, train_config
    )
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        out_dir / config["quadrate_model_path"],
    )

    # Plot training and validation losses
    fig = training.plot_losses(train_losses, val_losses)
    fig.savefig(out_dir / "losses.png")
    plt.close(fig)

    # Perform inference
    patch_size = get_patch_size(config)
    test_img = test_subject[tio.IMAGE][tio.DATA].squeeze().numpy()
    prediction = model.predict(
        net,
        test_subject,
        patch_size=patch_size,
        patch_overlap=(4, 4, 4),
        activation=model.activation_name(config),
        batch_size=config["batch_size"],
    )
    thresholded = prediction > 0.5
    prediction = metrics.largest_connected_component(thresholded)

    # Plot truth
    truth = test_subject[tio.LABEL][tio.DATA].squeeze().numpy()
    fig, _ = images_3d.plot_slices(test_img, truth)
    fig.savefig(out_dir / "test_truth.png")
    plt.close(fig)

    # some metrics
    dice = metrics.dice_score(truth, prediction)
    hausdorff = metrics.hausdorff_distance(truth, thresholded)
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(
            metrics.table([truth], [prediction], thresholded_metrics=True).to_markdown()
        )

    # Plot test subject
    fig, _ = images_3d.plot_slices(test_img, prediction)
    fig.suptitle(f"Dice: {100 * dice:.2f}%, Hausdorff: {100 * hausdorff:.2f}%")
    fig.savefig(out_dir / "test_inference.png")
    plt.close(fig)

    # Plot meshes with Hausdorff points indicated
    pred_mesh = mesh.cubic_mesh(thresholded)
    truth_mesh = mesh.cubic_mesh(truth)

    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))
    hausdorff_points = metrics.hausdorff_points(truth, thresholded)

    # Make projections of the meshes
    plot_meshes.projections(
        axes,
        pred_mesh,
        plot_kw={"alpha": 0.3, "color": "blue", "label": "Prediction"},
    )
    plot_meshes.projections(
        axes, truth_mesh, plot_kw={"alpha": 0.2, "color": "red", "label": "Truth"}
    )

    # Indicate Hausdorff distance
    x, y, z = zip(*hausdorff_points)
    for ax in axes:
        ax.plot(x, y, z, "rx-", markersize=4, label="Hausdorff distance")

    axes[0].legend(loc="upper right")

    fig.savefig(
        f"{out_dir}/overlaid_meshes.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def fine_tune(
    *, base_model: str, epochs: int, train_layers: list[int], train_all: bool, **kwargs
):
    """
    Read in the training data, load in the base model and fine tune it
    Make some plots of the loss, the inference on the testing data and output some metrics
    """
    config = util.userconf()
    out_dir = pathlib.Path(config["script_output"]) / "transfer_learning" / "quadrate"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_subjects, val_subjects, test_subjects = data.quadrate_data(config)
    quadrate_data = DataConfig(config, train_subjects, val_subjects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Either train or fine tune a model on a small dataset of quadrates"
    )

    # Separate CLI args for training and fine tuning
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser(
        "train", help="Train a model on a small dataset of quadrates"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=650,
        help="Number of epochs of the quadrate data to use",
    )
    train_parser.set_defaults(func=train)

    fine_tune_parser = subparsers.add_parser(
        "fine_tune",
        help="Fine tune a model on a small dataset of quadrates",
    )
    fine_tune_parser.add_argument("base_model", help="Base model to fine-tune")
    fine_tune_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    group = fine_tune_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train-layers",
        type=int,
        nargs="+",
        help="Layers to train, e.g. 0 1 2",
        choices=list(range(6)),
    )
    group.add_argument("--train-all", action="store_true", help="Train all layers")
    fine_tune_parser.set_defaults(func=fine_tune)

    args = parser.parse_args()
    args.func(**vars(args))
