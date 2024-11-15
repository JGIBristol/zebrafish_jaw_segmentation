"""
Create meshes and stuff

"""

import numpy as np
from stl import mesh
from skimage import measure


def cubic_mesh(segmentation: np.ndarray) -> mesh.Mesh:
    """
    Turn a segmentation into a cubic meshmesh

    :param segmentation: The segmentation to turn into a mesh, as a binary array
    :param threshold: The threshold to use for the marching cubes algorithm

    :returns: The mesh

    """
    if set(np.unique(segmentation)) - {0, 1}:
        raise ValueError(f"Segmentation must be binary: got {np.unique(segmentation)=}")

    # Marching cubes
    verts, faces, *_ = measure.marching_cubes(segmentation, level=0.5)

    # Save as STL
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    stl_mesh.vectors = verts[faces]

    return stl_mesh
