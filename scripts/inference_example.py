"""
Perform inference on an out-of-sample subject

"""

import pickle
import pathlib
import argparse

import torch
import tifffile
import numpy as np
from stl import mesh
import torchio as tio
from skimage import measure
import matplotlib.pyplot as plt

from fishjaw.util import files
from fishjaw.model import model, data
from fishjaw.images import transform, metrics
from fishjaw.visualisation import images_3d


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
        with open(
            str(files.script_out_dir() / "train_output" / "test_subject.pkl"), "rb"
        ) as f:
            return pickle.load(f)
    else:
        window_size = transform.window_size(config)

    # Create a subject from the chosen image
    # Read the chosen image
    img_n = args.subject
    img = _read_img(config, img_n)

    # Crop it to the jaw
    crop_lookup = {
        218: (1700, 396, 296),  # 24month wt wt dvl:gfp contrast enhance
        219: (1411, 344, 420),  # 24month wt wt dvl:gfp contrast enhance
        # 247: (1710, 431, 290),  # 14month het sp7 sp7+/-
        273: (1685, 221, 286),  # 9month het sp7 sp7 het
        274: (1413, 174, 240),  # 9month hom sp7 sp7 mut
        120: (1595, 251, 398),  # 10month wt giantin giantin sib
        37: (1746, 431, 405),  # 7month wt wt col2:mcherry
    }
    img = transform.crop(img, crop_lookup[img_n], window_size, centred=True)

    # Scale to [0, 1]
    img = data.ints2float(img)

    # Create a subject
    return _get_subject(img)


def _mesh_projections(stl_mesh: mesh.Mesh) -> plt.Figure:
    """
    Visualize the mesh from three different angles

    """
    vertices = stl_mesh.vectors.reshape(-1, 3)
    faces = np.arange(vertices.shape[0]).reshape(-1, 3)

    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))

    plot_kw = {"cmap": "bone_r", "edgecolor": "k", "lw": 0.05}
    # First subplot: view from the front
    axes[0].plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        **plot_kw,
    )
    axes[0].view_init(elev=0, azim=0)

    # Second subplot: view from the top
    axes[1].plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        **plot_kw,
    )
    axes[1].view_init(elev=90, azim=0)

    # Third subplot: view from the side
    axes[2].plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        **plot_kw,
    )
    axes[2].view_init(elev=0, azim=90)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.tight_layout()
    return fig


def _save_mesh(segmentation: np.ndarray, subject_name: str, threshold: float) -> None:
    """
    Turn a segmentation into a mesh and save it

    """
    # Marching cubes
    verts, faces, *_ = measure.marching_cubes(segmentation, level=threshold)

    # Save as STL
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    stl_mesh.vectors = verts[faces]

    stl_mesh.save(f"inference/{subject_name}_mesh_{threshold:.3f}.stl")

    # Save projections
    fig = _mesh_projections(stl_mesh)
    fig.savefig(f"inference/{subject_name}_mesh_{threshold:.3f}_projections.png")
    plt.close(fig)


def _make_plots(
    args: argparse.Namespace,
    net: torch.nn.Module,
    subject: tio.Subject,
    config: dict,
    activation: str,
) -> None:
    """
    Make the inference plots using a model and subject
    """
    # Perform inference
    prediction = model.predict(
        net,
        subject,
        patch_size=data.get_patch_size(config),
        patch_overlap=(4, 4, 4),
        activation=activation,
    )

    # Convert the image to a 3d numpy array - for plotting
    image = subject[tio.IMAGE][tio.DATA].squeeze().numpy()

    out_dir = files.script_out_dir() / "inference"
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
    plt.close(fig)

    # Save the mesh
    if args.mesh:
        for threshold in np.arange(0.1, 1, 0.1):
            _save_mesh(prediction, args.subject, threshold)


def _inference(args: argparse.Namespace, net: torch.nn.Module, config: dict) -> None:
    """
    Do the inference, save the plots

    """
    # Find which activation function to use from the config file
    # This assumes this was the same activation function used during training...
    activation = model.activation_name(config)

    # Either iterate over all subjects or just do the one
    if args.all:
        # Bad, should read these from a single place
        for subject in [273, 274, 218, 219, 120, 37]:
            print(f"Performing inference on subject {subject}")
            args.subject = subject
            _make_plots(args, net, _subject(config, args), config, activation)
    else:
        _make_plots(args, net, _subject(config, args), config, activation)


def main(args):
    """
    Load the model, read the chosen image and perform inference
    Save the output image

    """
    if args.subject == 247:
        raise RuntimeError("I think this one was in the training dataset...")

    # Load the model and training-time config
    with open(str(files.model_path()), "rb") as f:
        model_state: model.ModelState = pickle.load(f)

    config = model_state.config
    net = model_state.load_model(set_eval=True)

    net.to("cuda")

    _inference(args, net, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on an out-of-sample subject"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "subject",
        nargs="?",
        help="The subject to perform inference on",
        choices={247, 273, 274, 218, 219, 120, 37},
        type=int,
    )
    group.add_argument(
        "--test", help="Perform inference on the test data", action="store_true"
    )
    group.add_argument(
        "--all", help="Perform inference on all subjects", action="store_true"
    )

    parser.add_argument("--mesh", help="Save the mesh", action="store_true")

    main(parser.parse_args())
