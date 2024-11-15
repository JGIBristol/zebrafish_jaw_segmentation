"""
Functions to read in our data in a format that can be used for inference

"""

import pickle
import pathlib

import torch
import tifffile
import torchio as tio

from fishjaw.util import files
from fishjaw.images import transform
from fishjaw.model import data


def inference_subject(
    config: dict, img_n: int, crop_centre: tuple[int, int, int]
) -> tio.Subject:
    """
    Read the image of choice and turn it into a Subject

    :param config: configuration, as might be read from userconf.yml
    :param img_n: the image number to read - reads from Wahab's 3D tiff files
    :param crop_centre: the co-ordinate to crop around. Gets the crop size from the config

    :returns: the image as a torchio Subject

    """
    img = tifffile.imread(files.wahab_3d_tifs_dir(config) / f"ak_{img_n}.tif")

    img = transform.crop(img, crop_centre, transform.window_size(config), centred=True)

    # Scale to [0, 1]
    img = data.ints2float(img)

    # Add a channel dimension
    tensor = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)

    return tio.Subject(image=tio.Image(tensor=tensor, type=tio.INTENSITY))


def test_subject(model_path: str) -> tio.Subject:
    """
    Load the testing subject that was dumped when we trained the model

    :param model_path: the path to the model, as created by scripts/train_model.py.
                       You might get this from userconf["model_path"]

    :returns: the testing subject

    """
    with open(
        str(
            files.script_out_dir()
            / "train_output"
            / pathlib.Path(model_path).stem
            / "test_subject.pkl"
        ),
        "rb",
    ) as f:
        return pickle.load(f)
