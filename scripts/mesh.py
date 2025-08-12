"""
Example showing creation of a mesh from a binary (3d) tiff, and displaying it with matplotlib

"""

import tifffile
import trimesh
import matplotlib.pyplot as plt
from skimage import measure


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Choose the largest connected component of the mesh

    """
    # The first mesh is the largest connected component
    return mesh.split(only_watertight=True)[0]


def _tiff2mesh(tiff_path: str, out_path: str) -> None:
    """
    Read the tiff file, turn it into a triangular mesh and save to disk


    """
    # Read a binary volume
    volume = tifffile.imread(tiff_path)
    volume = volume > 0

    # Generate a mesh, convert it to triangles
    verts, faces, *_ = measure.marching_cubes(volume, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Pick the largest body
    largest = _largest_component(mesh)

    largest.export(out_path)

    return largest


def _visualize_mesh(
    mesh: trimesh.Trimesh, axes: tuple[plt.Axes, plt.Axes, plt.Axes]
) -> None:
    """
    Visualize the mesh from three different angles

    """
    ax1, ax2, ax3 = axes

    plot_kw = {"cmap": "bone_r", "edgecolor": "k", "lw": 0.05}

    # First subplot: view from the front
    ax1.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        **plot_kw,
    )
    ax1.view_init(elev=0, azim=0)

    # Second subplot: view from the top
    ax2.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        **plot_kw,
    )
    ax2.view_init(elev=90, azim=0)

    # Third subplot: view from the side
    ax3.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        **plot_kw,
    )
    ax3.view_init(elev=0, azim=90)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


def main():
    """
    Read a tiff file and turn it into a triangular mesh

    """
    # Ground truth
    truth_mesh = _tiff2mesh("../fish_test/mask_dumps/491.tif", "truth.stl")

    # Model prediction
    pred_mesh = _tiff2mesh("../fish_test/tmp_pred.tif", "pred.stl")

    # Plot comparisons
    fig, axes = plt.subplots(3, 2, subplot_kw={"projection": "3d"}, figsize=(10, 15))

    _visualize_mesh(truth_mesh, axes[:, 0])
    _visualize_mesh(pred_mesh, axes[:, 1])

    axes[0, 0].set_title("Ground truth")
    axes[0, 1].set_title("Model prediction")

    fig.tight_layout()
    fig.savefig("meshes.png")


if __name__ == "__main__":
    main()
