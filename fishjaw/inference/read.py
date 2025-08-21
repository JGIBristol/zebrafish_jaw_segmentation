"""
Functions to read in our data in a format that can be used for inference

"""

import pickle
import pathlib
import textwrap
from functools import cache
from dataclasses import dataclass

import torch
import pydicom
import tifffile
import numpy as np
import pandas as pd
import torchio as tio

from fishjaw.util import files
from fishjaw.images import transform
from fishjaw.model import data


@dataclass
class Metadata:
    """
    The interesting/important metadata for a fish sample
    """

    n: int
    """ Fish number, using the new n (i.e. n not old_n) convention """
    age: int
    """ Age in months """
    genotype: str
    """ Genotype, e.g. wt/hom/het"""
    strain: str
    """ e.g. chst11, runx2, wt"""
    name: str
    """ e.g. fli:gfp, wt, dot1 +/-"""
    length: float
    """ Not sure what this is: possibly fish length in mm"""
    voxel_volume: float
    """ Volume of each voxel; not sure of the units, possibly mm^3"""
    comments: str
    """Any other comments - importantly sometimes contains info about contrast enhancement"""

    def __str__(self):
        age = f"{self.age} months" if self.age >= 0 else "unknown age"
        return (
            f"N={self.n}: {self.genotype} {self.strain} {self.name} "
            f"({age})\n{textwrap.fill(self.comments)}".strip()
        )


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
        5: (1768, 281, 374),  # 24 month wt,wt
        6: (1751, 476, 476),  # 24 month wt,wt
        7: (1600, 415, 274),  # 24 month wt,wt
        344: (1626, 357, 397),  # 6month wt,wt
        345: (1820, 322, 344),  # 6month wt,wt
        346: (1558, 272, 307),  # 6month wt,wt
        317: (1430, 378, 320),  # 6month wt,tert
        318: (1332, 346, 401),  # 6month wt,tert
        319: (1335, 332, 264),  # 6month wt,tert
        415: (1733, 339, 309),  # 24month wt,wt
        416: (1605, 358, 199),  # 24month wt,wt
        417: (1655, 323, 374),  # 24month wt,wt
    }


def _ct_scan_array(config: dict, img_n: int) -> np.ndarray:
    """
    Get the CT scan of choice as a greyscale numpy array.

    This will be read from the 3D TIFS if possible, otherwise
    will be read from the DICOMs.

    :param config: configuration, as might be read from userconf.yml
    :param img_n: the image number to read - reads from Wahab's 3D tiff files

    :returns: the image

    """
    try:
        img = tifffile.imread(files.wahab_3d_tifs_dir(config) / f"{img_n}.tif")
    except FileNotFoundError:
        dicom = pydicom.dcmread(files.wahab_dicoms_dir(config) / f"ak_{img_n}.dcm")
        # Assume that this is the convention; its the default...
        dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = dicom.pixel_array

    return img


def cropped_img(config: dict, img_n: int) -> np.ndarray:
    """
    Read + crop the required image

    """
    return transform.crop(
        _ct_scan_array(config, img_n),
        crop_lookup()[img_n],
        transform.window_size(config),
        centred=True,
    )


def inference_subject(config: dict, img_n: int) -> tio.Subject:
    """
    Read the image of choice and turn it into a Subject, cropping it according
    to `read.crop_lookup`

    :param config: configuration, as might be read from userconf.yml
    :param img_n: the image number to read - reads from Wahab's 3D tiff files

    :returns: the image as a torchio Subject

    """
    img = cropped_img(config, img_n)

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


@cache
def mastersheet() -> pd.DataFrame:
    """
    Metadata mastersheet, in a more useful format
    """
    retval = files._mastersheet()

    retval = retval[
        [
            "n",
            "age",
            "genotype",
            "strain",
            "name",
            "VoxelSizeX",
            "VoxelSizeY",
            "VoxelSizeZ",
            "length",
            "Comments",
        ]
    ]

    # Fill NaN ages with -1 and empty comments with empty str
    retval.loc[:, "age"] = retval["age"].fillna(-1)
    retval.loc[:, "Comments"] = retval["Comments"].fillna("")

    # Convert datatypes
    retval = retval.astype(
        {
            "n": int,
            "age": int,
            "length": float,
            **{col: float for col in ["VoxelSizeX", "VoxelSizeY", "VoxelSizeZ"]},
        }
    )

    retval.set_index("n", inplace=True)
    return retval


def fish_number(path: pathlib.Path) -> int:
    """
    Get the fish number from a filename, assuming the file follows the
    "ak_<n>.<ext>" pattern

    :param path: the path to the file

    :returns: the fish number
    """
    return int(path.stem.split("_")[1])


def metadata(fish_n: int) -> Metadata:
    """
    Get the metadata for one fish

    :param fish_n: fish number, using the "n" (i.e. new, not "old_n") convention
    :returns: Metadata

    """
    # Get a pd.Series representing the right stuff
    df = mastersheet().loc[fish_n]

    # Turn it into a metadata object
    return Metadata(
        n=fish_n,
        age=df["age"],
        genotype=df["genotype"],
        strain=df["strain"],
        name=df["name"],
        voxel_volume=df["VoxelSizeX"] * df["VoxelSizeY"] * df["VoxelSizeZ"],
        length=df["length"],
        comments=df["Comments"],
    )
