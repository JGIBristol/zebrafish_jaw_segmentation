"""
Perform inference on an out-of-sample subject

"""

import argparse

import torch
import tifffile
import numpy as np

from fishjaw.util import files
from fishjaw.model import model


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


def _read_img(subject: int) -> np.ndarray:
    """
    Read the chosen image

    """
    path = files.wahab_3d_tifs_dir() / f"ak_{subject}.tif"
    return tifffile.imread(path)


def _infer(model: torch.nn.Module, img: np.ndarray) -> np.ndarray:
    """
    Perform inference on an image

    """


def _save_img(img: np.ndarray, prediction: np.ndarray):
    """
    Save the output image

    """


def main(args):
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    model = _load_model()

    # Read the chosen image
    img_n = args.subject
    img = _read_img(img_n)
    print(img)

    # Perform inference
    prediction = _infer(model, img)

    # Save the output image
    _save_img(img, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )
    parser.add_argument(
        "subject", help="The subject to perform inference on", choices={273}, type=int
    )
    main(parser.parse_args())
