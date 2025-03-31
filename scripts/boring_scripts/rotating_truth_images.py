"""
Make an image stack showing the jaw rotating for the truth data labelled
by Tahlia, Harry and Felix, and a model.

you can turn them into videos with e.g.
for filename in boring_script_output/rotating_meshes/*; do n=`basename $filename`; ffmpeg -framerate 12 -pattern_type glob -i "boring_script_output/rotating_meshes/${n}/*.png" -c:v libx264 -pix_fmt yuv420p boring_script_output/rotating_meshes/${n}.mp4; done

"""

import pathlib
import argparse

import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from fishjaw.util import files, util
from fishjaw.model import model, data
from fishjaw.images import metrics, transform
from fishjaw.inference import read


def rotating_plots(mask: np.ndarray, out_dir: pathlib.Path) -> None:
    """
    Save an lots of images of a rotating mesh, which we can then
    turn into a gif

    """
    plt.switch_backend("agg")

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    def plot_proj(enum_angles):
        """Helper fcn to change the rotation of a plot"""
        i, angles = enum_angles
        axis.view_init(*angles)
        fig.savefig(f"{out_dir}/mesh_{i:03}.png")
        plt.close(fig)
        return i

    # Make a scatter plot of the mask
    fig = plt.figure()
    axis = fig.add_subplot(projection="3d")
    co_ords = np.argwhere(mask > 0.5)
    axis.scatter(
        co_ords[:, 0],
        co_ords[:, 1],
        co_ords[:, 2],
        c=co_ords[:, 2],
        cmap="copper",
        alpha=0.5,
    )
    axis.axis("off")

    # Plot the mesh at various angles
    num_frames = 108
    azimuths = np.linspace(-90, 270, num_frames, endpoint=False)
    elevations = list(np.linspace(-90, 90, num_frames // 2)) + list(
        np.linspace(90, -90, num_frames // 2)
    )
    rolls = np.linspace(0, 360, num_frames, endpoint=False)

    angles = list(enumerate(zip(azimuths, elevations, rolls)))

    for angle in tqdm(angles):
        plot_proj(angle)


def _inference(model_name: str) -> np.ndarray:
    """
    Do the inference, save the plots

    """
    # Load the model and training-time config
    model_state = model.load_model(model_name)

    config = model_state.config
    net = model_state.load_model(set_eval=True)
    net.to("cuda")

    prediction = model.predict(
        net,
        read.inference_subject(config, 97),
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=model.activation_name(config),
    )

    # Threshold the segmentation
    prediction = (prediction > 0.5).astype(np.uint8)

    return metrics.largest_connected_component(prediction)


def main(model_name: str, skip_human: bool):
    """
    Read the jaws in, make rotating images of the truth data
    """
    print("loading human segmentations")
    config = util.userconf()
    seg_dir = (
        files.rdsf_dir(config)
        / "1Felix and Rich make models"
        / "Human validation STL and results"
    )
    felix = tifffile.imread(
        seg_dir / "felix take2" / "ak_97-FBowers_complete.labels.tif"
    )
    harry = tifffile.imread(seg_dir / "Harry" / "ak_97.tif.labels.tif")
    tahlia = tifffile.imread(seg_dir / "Tahlia" / "tpollock_97_avizo.labels.tif")

    print("Performing inference")
    inference = _inference(model_name)

    # Crop them to the same size as the model's output
    print("Cropping")
    felix, harry, tahlia = (
        transform.crop(
            x, read.crop_lookup()[97], transform.window_size(config), centred=True
        )
        for x in (felix, harry, tahlia)
    )

    out_dir = files.boring_script_out_dir() / "rotating_meshes"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    model_name, = model_name.split("*.pkl")
    rotating_plots(inference, out_dir / model_name)
    if not skip_human:
        rotating_plots(tahlia, out_dir / "tahlia")
        rotating_plots(felix, out_dir / "felix")
        rotating_plots(harry, out_dir / "harry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make rotating images of Tahlia, Harry, Felix and the model's jaws"
    )

    parser.add_argument("model_name", type=str, help="The name of the model to use")
    parser.add_argument(
        "--skip_human",
        action="store_true",
        help="Skip the human segmentations and only create the inference files",
    )

    main(**vars(parser.parse_args()))
