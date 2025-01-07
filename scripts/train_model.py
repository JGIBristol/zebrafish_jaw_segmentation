"""
Train a model to segment the jawbone from labelled DICOM images

"""

import pickle
import pathlib
import argparse

import torch
import torchio as tio
import matplotlib.pyplot as plt

from fishjaw.util import files, util
from fishjaw.model import data, model
from fishjaw.visualisation import images_3d, training


def _plot_example(out_dir: pathlib.Path, batch: dict[str, torch.Tensor]):
    """
    Plot an example of the training data

    """
    img = batch[tio.IMAGE][tio.DATA][0, 0].numpy()
    label = batch[tio.LABEL][tio.DATA][0, 0].numpy()

    fig, _ = images_3d.plot_slices(img, label)

    fig.savefig(f"{out_dir}/train_example.png")


def train_model(
    config: dict,
    data_config: data.DataConfig,
    out_dir: pathlib.Path,
) -> tuple[
    tuple[torch.nn.Module, list[list[float]], list[list[float]], torch.optim.Optimizer]
]:
    """
    Create a model, train and return it

    Returns the model, the training losses and the validation losses, and the optimiser

    """
    # Create a model and optimiser
    net = model.model(config["model_params"])

    device = config["device"]
    net = net.to(device)

    optimiser = model.optimiser(config, net)

    # Plot an example of the training data (which has been augmented)
    _plot_example(out_dir, next(iter(data_config.train_data)))

    # Define loss function
    loss = model.lossfn(config)

    train_config = model.TrainingConfig(
        device,
        config["epochs"],
        torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=config["lr_lambda"]),
    )
    return (
        model.train(net, optimiser, loss, data_config, train_config),
        optimiser,
    )


def main():
    """
    Get the right data, train the model and create some outputs

    """
    config = util.userconf()

    # If the model is already cached, don't train it again
    model_path = files.model_path(config)
    if model_path.is_file():
        raise FileExistsError(f"Model already exists at {model_path}")
    print(f"Training model to save at {model_path}")

    torch.manual_seed(config["torch_seed"])

    # Find the activation - we'll need this for inference
    activation = model.activation_name(config)

    # Read the data from disk (from the DICOMs created by create_dicoms.py)
    train_subjects, val_subjects, test_subject = data.read_dicoms_from_disk(config)
    data_config = data.DataConfig(config, train_subjects, val_subjects)

    # Save the testing subject
    output_dir = (
        pathlib.Path(util.config()["script_output"]) / "train_output" / model_path.stem
    )
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    print(f"Saving outputs to {output_dir}")

    with open(output_dir / "test_subject.pkl", "wb") as f:
        pickle.dump(test_subject, f)

    (net, train_losses, val_losses), optimiser = train_model(
        config, data_config, output_dir
    )

    torch.cuda.empty_cache()

    # Save the model
    with open(str(model_path), "wb") as f:
        pickle.dump(
            model.ModelState(net.state_dict(), optimiser.state_dict(), config),
            f,
        )

    # Plot the loss
    fig = training.plot_losses(train_losses, val_losses)
    fig.savefig(str(output_dir / "loss.png"))
    plt.close(fig)

    # Plot the testing image
    fig = images_3d.plot_inference(
        net,
        test_subject,
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=activation,
        batch_size=config["batch_size"]
    )
    fig.savefig(str(output_dir / "test_pred.png"))
    plt.close(fig)

    # Plot the ground truth for this image
    fig, _ = images_3d.plot_subject(test_subject)
    fig.savefig(str(output_dir / "test_truth.png"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to segment the jawbone")
    main(**vars(parser.parse_args()))
