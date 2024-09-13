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
from fishjaw.model import model
from fishjaw.images import io, transform
from fishjaw.visualisation import images_3d


def _load_model() -> torch.nn.Module:
    """
    Load the model from disk

    """
    net = model.monai_unet()

    # Load the state dict
    path = files.model_path()
    state_dict = torch.load(path)

    net.load_state_dict(state_dict["model"])
    net.eval()

    return net


def _read_img(img_n: int) -> np.ndarray:
    """
    Read the chosen image

    """
    path = files.wahab_3d_tifs_dir() / f"ak_{img_n}.tif"
    return tifffile.imread(path)


def _get_subject(img: np.ndarray) -> tio.Subject:
    """
    Convert the image into a subject

    """
    tensor = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
    return tio.Subject(image=tio.Image(tensor=tensor, type=tio.INTENSITY))


def _subject(args: argparse.Namespace) -> tio.Subject:
    """
    Either read the image of choice and turn it into a Subject, or load the testing subject

    """
    # Load the testing subject
    if args.test:
        with open("train_output/test_subject.pkl", "rb") as f:
            return pickle.load(f)

    # Create a subject from the chosen image
    # Read the chosen image
    img_n = args.subject
    img = _read_img(img_n)

    # Crop it to the jaw
    crop_lookup = {
        247: (1710, 431, 290),
        273: (1685, 221, 286),
        274: (1413, 174, 240),
    }
    img = transform.crop(img, crop_lookup[img_n])

    # Scale to [0, 1]
    img = img / 65535

    # Create a subject
    return _get_subject(img)


def main(args):
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    subject = _subject(args)

    # Load the model
    model = _load_model()
    model.to("cuda")

    # Find which activation function to use from the config file
    # This assumes this was the same activation function used during training...
    config = util.userconf()
    if config["loss_options"].get("softmax", False):
        activation = "softmax"
    elif config["loss_options"].get("sigmoid", False):
        activation = "sigmoid"
    else:
        raise ValueError("No activation found")

    # Perform inference
    fig = images_3d.plot_inference(
        model,
        subject,
        patch_size=io.patch_size(),
        patch_overlap=(4, 4, 4),
        activation=activation,
    )

    # Save the output image
    out_dir = pathlib.Path("inference/")
    if not out_dir.exists():
        out_dir.mkdir()
    out_path = (
        out_dir / f"inference_{args.subject}.png"
        if args.subject
        else out_dir / "test.png"
    )
    fig.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subject",
        help="The subject to perform inference on",
        choices={247, 273, 274},
        type=int,
    )
    group.add_argument(
        "--test", help="Perform inference on the test data", action="store_true"
    )

    main(parser.parse_args())
