import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

def Den(means:np.ndarray):
    tri = Delaunay(means)

    faces = []
    for tet in tri.simplices:
        for i in range(4):
            face = np.delete(tet, i)
            faces.append(face)

    faces = np.array(faces)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(means)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    o3d.io.write_triangle_mesh("delaunay_surface_mesh.ply", mesh)