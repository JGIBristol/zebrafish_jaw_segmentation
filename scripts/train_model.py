"""
Train a model to segment the jawbone from labelled DICOM images

"""

import pathlib
import warnings
import argparse
from typing import Union

import torch
import numpy as np
import torchio as tio
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.losses import TverskyLoss

from fishjaw.util import files, util
from fishjaw.model import data, model
from fishjaw.visualisation import images_3d
from fishjaw.images import transform


def centre(dicom_path: pathlib.Path) -> tuple[int, int, int]:
    """
    Get the centre of the jaw for a given fish

    """
    n = int(dicom_path.stem.split("_", maxsplit=1)[-1])
    return transform.centre(n)


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


def plot_test_image(
    net: torch.nn.Module, sampler: tio.GridSampler, output_dir: pathlib.Path
) -> plt.Figure:
    """
    Plot the test image

    """
    agg_kw = {"sampler": sampler, "overlap_mode": "hann"}
    pred_aggregator = tio.inference.GridAggregator(**agg_kw)
    img_aggregator = tio.inference.GridAggregator(**agg_kw)
    truth_aggregator = tio.inference.GridAggregator(**agg_kw)

    # Our image is smaller than 1 batch, so we can just do it all at once
    batch = []
    locations = []
    truth = []
    for patch in sampler:
        gpu_patch = patch[tio.IMAGE][tio.DATA].unsqueeze(0)
        truth_patch = patch[tio.LABEL][tio.DATA].unsqueeze(0)

        locations.append(patch[tio.LOCATION])
        batch.append(gpu_patch)
        truth.append(truth_patch)

    device = next(net.parameters()).device

    batch_tensor = torch.cat(batch, dim=0).to(device)
    truth_tensor = torch.cat(truth, dim=0).to(device)
    locations_tensor = torch.stack(locations)
    patch_pred = net(batch_tensor).to("cpu").detach()

    pred_aggregator.add_batch(patch_pred, locations=locations_tensor)
    img_aggregator.add_batch(batch_tensor, locations=locations_tensor)
    truth_aggregator.add_batch(truth_tensor, locations=locations_tensor)

    prediction = pred_aggregator.get_output_tensor()[1].numpy()
    image = img_aggregator.get_output_tensor()[0].numpy()
    truth = truth_aggregator.get_output_tensor()[0].numpy()

    fig, _ = images_3d.plot_slices(image, prediction)
    fig.suptitle("Model Prediction")
    fig.tight_layout()
    fig.savefig(str(output_dir / "prediction.png"))

    fig, _ = images_3d.plot_slices(image, truth)
    fig.suptitle("Truth")
    fig.tight_layout()
    fig.savefig(str(output_dir / "truth.png"))


def main(*, pretrained: Union[str, None]) -> None:
    """
    Create a model, train it, then save it

    """
    # Check that the pretrained state dict is valid
    state_dict = torch.load(pretrained) if pretrained is not None else None

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
    if state_dict is not None:
        net.load_state_dict(state_dict["model_state_dict"])
        optimiser.load_state_dict(state_dict["optimizer_state_dict"])
        print(f"Loaded pretrained model, trained for {state_dict['epoch']} epochs")

    output_dir = pathlib.Path("train_output")
    if not output_dir.is_dir():
        output_dir.mkdir()

    # Read in data
    dicom_paths = list(files.dicom_dir().glob("*.dcm"))

    # Convert to subjects
    subjects = [
        data.subject(path, centre=centre(path))
        for path in tqdm(dicom_paths, desc="Reading DICOMs")
    ]

    # Define some random transforms
    transforms = tio.Compose(
        [
            # tio.RandomFlip(axes=(0), flip_probability=0.5),
            # tio.RandomAffine(
            #     p=1,
            #     degrees=10,
            #     scales=0.5,
            # ),
        ]
    )

    # Choose some indices to act as train, validation and test
    rng = np.random.default_rng(seed=0)
    indices = np.arange(len(subjects))
    rng.shuffle(indices)
    train_idx, val_idx, test_idx = np.split(
        indices, [int(0.95 * len(indices)), len(indices) - 1]
    )
    assert len(test_idx) == 1
    test_idx = test_idx[0]

    print(f"Train: {len(train_idx)=}")
    print(f"Val: {val_idx=}")
    print(f"Test: {test_idx=}")

    train_subjects = tio.SubjectsDataset(
        [subjects[i] for i in train_idx], transform=transforms
    )
    val_subjects = tio.SubjectsDataset([subjects[i] for i in val_idx])

    # Convert to dataloaders
    uconf = util.userconf()
    patch_size = (128, 128, 128)
    batch_size = uconf["batch_size"]
    train_loader = data.train_val_loader(
        train_subjects, shuffle=True, patch_size=patch_size, batch_size=batch_size
    )
    val_loader = data.train_val_loader(
        val_subjects, shuffle=False, patch_size=patch_size, batch_size=batch_size
    )

    patch_overlap = (4, 4, 4)
    test_sampler = tio.GridSampler(
        subjects[test_idx], patch_size, patch_overlap=patch_overlap
    )

    # Plot an example of the training data
    example_data = next(iter(train_loader))
    fig, _ = images_3d.plot_slices(
        example_data[tio.IMAGE][tio.DATA][0].squeeze().numpy(),
        mask=example_data[tio.LABEL][tio.DATA][0].squeeze().numpy(),
    )
    fig.savefig(str(output_dir / "example_data.png"))

    # Define loss function
    loss = TverskyLoss(include_background=False, to_onehot_y=True, alpha=0.5)

    # Train the model
    net, train_losses, val_losses = model.train(
        net,
        optimiser,
        loss,
        train_loader,
        val_loader,
        device=device,
        epochs=util.userconf()["epochs"],
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=uconf["lr_lambda"]),
        checkpoint=True,
    )

    # Plot the loss
    fig = plot_losses(train_losses, val_losses)
    fig.savefig(str(output_dir / "loss.png"))

    # Plot the testing image
    plot_test_image(net, test_sampler, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to segment the jawbone")

    default_weights = "data/starting_weights.pth"
    parser.add_argument(
        "--pretrained",
        nargs="?",
        type=str,
        const=default_weights,
        help=f"""Whether to load a pretrained model at the provided path.
                 Defaults to {default_weights} if no path is provided""",
    )

    main(**vars(parser.parse_args()))
