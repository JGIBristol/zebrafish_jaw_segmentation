"""
Perform inference on an out-of-sample subject

"""

import pickle
import pathlib
import argparse

import torch
import tifffile
import numpy as np
import torchio as tio

from fishjaw.util import files, util
from fishjaw.model import model, data
from fishjaw.images import io, transform, metrics
from fishjaw.visualisation import images_3d


def _load_model(config: dict) -> torch.nn.Module:
    """
    Load the model from disk

    """
    net = model.monai_unet(params=model.model_params(config["model_params"]))

    # Load the state dict
    path = files.model_path()
    state_dict = torch.load(path)

    net.load_state_dict(state_dict["model"])
    net.eval()

    return net


def _read_img(config: dict, img_n: int) -> np.ndarray:
    """
    Read the chosen image

    """
    path = files.wahab_3d_tifs_dir(config) / f"ak_{img_n}.tif"
    return tifffile.imread(path)


def _get_subject(img: np.ndarray) -> tio.Subject:
    """
    Convert the image into a subject

    """
    tensor = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
    return tio.Subject(image=tio.Image(tensor=tensor, type=tio.INTENSITY))


def _subject(config: dict, args: argparse.Namespace) -> tio.Subject:
    """
    Either read the image of choice and turn it into a Subject, or load the testing subject

    """
    # Load the testing subject
    if args.test:
        with open("train_output/test_subject.pkl", "rb") as f:
            return pickle.load(f)
    else:
        window_size = transform.window_size(config["window_size"])

    # Create a subject from the chosen image
    # Read the chosen image
    img_n = args.subject
    img = _read_img(config, img_n)

    # Crop it to the jaw
    crop_lookup = {
        247: (1710, 431, 290),
        273: (1685, 221, 286),
        274: (1413, 174, 240),
    }
    img = transform.crop(img, crop_lookup[img_n], window_size)

    # Scale to [0, 1]
    img = data.ints2float(img)

    # Create a subject
    return _get_subject(img)


def main(args):
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    config = util.userconf()
    subject = _subject(config, args)

    # Load the model
    net = _load_model(config)
    net.to("cuda")

    # Find which activation function to use from the config file
    # This assumes this was the same activation function used during training...
    if config["loss_options"].get("softmax", False):
        activation = "softmax"
    elif config["loss_options"].get("sigmoid", False):
        activation = "sigmoid"
    else:
        raise ValueError("No activation found")

    # Perform inference
    prediction = model.predict(
        net,
        subject,
        patch_size=data.patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=activation,
    )

    # Convert the image to a 3d numpy array - for plotting
    image = subject[tio.IMAGE][tio.DATA].squeeze().numpy()

    out_dir = pathlib.Path("inference/")
    if not out_dir.exists():
        out_dir.mkdir()

    # Save the image and prediction as tiffs
    prefix = args.subject if args.subject else "test"
    tifffile.imwrite(out_dir / f"{prefix}_image.tif", image)
    tifffile.imwrite(out_dir / f"{prefix}_prediction.tif", prediction)

    # Save the output image and prediction as slices
    fig, _ = images_3d.plot_slices(image, prediction)

    # If we're using the test data, we have access to the ground truth so can
    # work out the Dice score and stick it in the plot too
    if args.test:
        truth = subject[tio.LABEL][tio.DATA].squeeze().numpy()
        dice = metrics.dice_score(truth, prediction)
        fig.suptitle(f"Dice: {dice:.3f}", y=0.99)

        # We might as well save the truth as a tiff too
        tifffile.imwrite(out_dir / f"{prefix}_truth.tif", truth)
    else:
        fig.suptitle(f"Inference: ID {args.subject}", y=0.99)

    fig.savefig(out_dir / f"{prefix}_slices.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "subject",
        nargs="?",
        help="The subject to perform inference on",
        choices={247, 273, 274},
        type=int,
    )
    group.add_argument(
        "--test", help="Perform inference on the test data", action="store_true"
    )

    main(parser.parse_args())
