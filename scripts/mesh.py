"""
Example showing creation of a mesh from a binary (3d) tiff, and displaying it with matplotlib

"""

from skimage import measure
import tifffile
import trimesh


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


def main():
    """
    Read a tiff file and turn it into a triangular mesh

    """
    # Ground truth
    _tiff2mesh("../fish_test/mask_dumps/491.tif", "truth.stl")

    # Model prediction
    _tiff2mesh("../fish_test/tmp_pred.tif", "pred.stl")


if __name__ == "__main__":
    main()
