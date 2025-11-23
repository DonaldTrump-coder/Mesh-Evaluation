import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

def Den(means:np.ndarray):
    tri = Delaunay(means)

    faces = []
    for tet in tri.simplices:
        # 4 个顶点的 tetrahedron 有 4 个三角面
        for i in range(4):
            face = np.delete(tet, i)
            faces.append(face)

    faces = np.array(faces)

    # 构建 Open3D 网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(means)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # 保存
    o3d.io.write_triangle_mesh("delaunay_surface_mesh.ply", mesh)