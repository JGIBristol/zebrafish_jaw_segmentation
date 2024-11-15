"""
Functions to read in our data in a format that can be used for inference

"""

import torch
import tifffile
import torchio as tio

from fishjaw.util import files
from fishjaw.images import transform


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

    img = transform.crop(img, crop_centre, config["crop_size"])

    # Add a channel dimension
    tensor = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)

    return tio.Subject(image=tio.Image(tensor=tensor, type=tio.INTENSITY))
