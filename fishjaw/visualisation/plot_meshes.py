"""
Visualise meshes

"""

import stl
import numpy as np
import matplotlib.pyplot as plt


def projections(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    mesh: stl.Mesh,
    *,
    plot_kw: dict | None = None,
) -> None:
    """
    Plot projections of the mesh from three different angles

    :param axes: Three axes to plot on
    :param mesh: The mesh to plot
    :param plot_kw: extra keyword arguments for the plot_trisurf function

    """
    if len(axes) != 3:
        raise ValueError(f"Expected 3 axes, got {len(axes)}")

    vertices = mesh.vectors.reshape(-1, 3)
    faces = np.arange(vertices.shape[0]).reshape(-1, 3)

    # Don't overwrite values if they've been provided in plot_kw
    plot_kw = plot_kw or {}
    plot_kw = {
        "cmap": plot_kw.get("cmap", "cividis_r"),
        "edgecolor": plot_kw.get("edgecolor", "k"),
        "lw": plot_kw.get("lw", 0.05),
        **plot_kw,
    }

    # First subplot: view from the front
    for axis, elev, azim in zip(axes, [0, 90, 0], [0, 0, 90]):
        axis.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, **plot_kw
        )
        axis.view_init(elev=elev, azim=azim)

        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
