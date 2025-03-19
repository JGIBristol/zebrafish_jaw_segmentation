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


def crop_lookup() -> dict[int, tuple[int, int, int]]:
    """
    Mapping from image number to crop centre, which I found by eye

    :returns: the crop centres for each image
    """
    return {
        218: (1700, 296, 396),  # 24month wt wt dvl:gfp contrast enhance
        219: (1411, 420, 344),  # 24month wt wt dvl:gfp contrast enhance
        # 247: (1710, 431, 290),  # 14month het sp7 sp7+/-
        273: (1685, 286, 221),  # 9month het sp7 sp7 het
        274: (1413, 240, 174),  # 9month hom sp7 sp7 mut
        120: (1595, 398, 251),  # 10month wt giantin giantin sib
        37: (1746, 405, 431),  # 7month wt wt col2:mcherry
        97: (1435, 174, 269),  # 36 month wt wt wnt:gfp col2a1:mch
        5: (1768, 281, 374),  # wt,wt
        6: (1751, 476, 476),  # wt,wt
        7: (1600, 415, 274),  # wt,wt
        344: (1626, 357, 397),  # wt,wt
        345: (0, 0, 0),  # wt,wt
        346: (0, 0, 0),  # wt,wt
        317: (0, 0, 0),  # wt,tert
        318: (0, 0, 0),  # wt,tert
        319: (0, 0, 0),  # wt,tert
        414: (0, 0, 0),  # wt,wt
        415: (0, 0, 0),  # wt,wt
        416: (0, 0, 0),  # wt,wt
    }


def inference_subject(config: dict, img_n: int) -> tio.Subject:
    """
    Read the image of choice and turn it into a Subject, cropping it according
    to `read.crop_lookup`

    :param config: configuration, as might be read from userconf.yml
    :param img_n: the image number to read - reads from Wahab's 3D tiff files

    :returns: the image as a torchio Subject

    """
    img = tifffile.imread(files.wahab_3d_tifs_dir(config) / f"{img_n}.tif")

    img = transform.crop(
        img, crop_lookup()[img_n], transform.window_size(config), centred=True
    )

    # Scale to [0, 1]
    img = data.ints2float(img)

    # Add a channel dimension
    tensor = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)

    return tio.Subject(image=tio.Image(tensor=tensor, type=tio.INTENSITY))


def test_subject(model_name: str) -> tio.Subject:
    """
    Load the testing subject that was dumped when we trained the model

    :param model_name: the path to the model, as created by scripts/train_model.py.
                       You might get this from userconf["model_path"] - e.g. "with_attention.pkl

    :returns: the testing subject

    """
    if not model_name.endswith(".pkl"):
        # This isn't technically a problem, but it's likely to be a mistake
        # so let's raise an error because its likely that the wrong model
        # name has been specified. It should be something like "my_model.pkl"
        raise ValueError(
            f"model_name should be name of a pickled model, not {model_name}"
        )

    with open(
        str(
            files.script_out_dir()
            / "train_output"
            / pathlib.Path(model_name).stem
            / "test_subject.pkl"
        ),
        "rb",
    ) as f:
        return pickle.load(f)
