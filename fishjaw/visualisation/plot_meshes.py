"""
Visualise meshes

"""
from typing import Any

import stl
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def projections(
    axes: tuple[
        matplotlib.axes.Axes,
        matplotlib.axes.Axes,
        matplotlib.axes.Axes,
    ],
    mesh: stl.Mesh,
    *,
    plot_kw: dict[str, Any] | None = None,
) -> None:
    """
    Plot projections of the mesh from three different angles

    :param axes: Three axes to plot on
    :param mesh: The mesh to plot
    :param plot_kw: extra keyword arguments for the plot_trisurf function

    """
    if len(axes) != 3:
        raise ValueError(f"Expected 3 axes, got {len(axes)}")

    plot_kw = plot_kw or {}

    # Check we haven't specified both a color and a cmap
    if "color" in plot_kw and "cmap" in plot_kw:
        raise ValueError(f"Cannot specify both a color and a cmap:\n{plot_kw}")

    vertices = mesh.vectors.reshape(-1, 3)
    faces = np.arange(vertices.shape[0]).reshape(-1, 3)

    # Deal with default plot arguments
    plot_kw["cmap"] = (
        plot_kw.get("cmap", "cividis_r") if "color" not in plot_kw else None
    )
    plot_kw = {
        "edgecolor": plot_kw.get("edgecolor", "none"),
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
