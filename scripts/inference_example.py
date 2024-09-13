"""
Perform inference on an out-of-sample subject

"""

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


def main(args):
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
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

    # Create a subject
    subject = _get_subject(img)

    # Load the model
    model = _load_model()
    model.to("cuda")

    # Perform inference
    config = util.userconf()
    if config["loss_options"].get("softmax", False):
        activation = "softmax"
    elif config["loss_options"].get("sigmoid", False):
        activation = "sigmoid"
    else:
        raise ValueError("No activation found")

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
    fig.savefig(out_dir / f"inference_{img_n}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )
    parser.add_argument(
        "subject",
        help="The subject to perform inference on",
        choices={247, 273, 274},
        type=int,
    )
    main(parser.parse_args())
