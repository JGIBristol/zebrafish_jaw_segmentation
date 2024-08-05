"""
Example showing creation of a mesh from a binary (3d) tiff

"""

from skimage import measure
import tifffile
import trimesh


def main():
    """
    Read a tiff file and turn it into a triangular mesh

    """
    # Read a binary volume
    volume = tifffile.imread("../fish_test/tmp_pred.tif")
    volume = volume > 0

    # Generate a mesh, convert it to triangles and save
    verts, faces, *_ = measure.marching_cubes(volume, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export("output_mesh.stl")


if __name__ == "__main__":
    main()
