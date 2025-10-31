import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

def Den(means:np.ndarray):
    tri = Delaunay(means)

    faces = set()
    for tet in tri.simplices:
        for i in range(4):
            face = tuple(sorted(np.delete(tet, i)))  # 删除一个顶点 -> 三角面
            if face in faces:
                faces.remove(face)  # 内部面出现两次，删掉
            else:
                faces.add(face)     # 外部面只出现一次

    faces = np.array(list(faces))

    # 构建 Open3D 网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(means)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # 可视化
    o3d.visualization.draw_geometries([mesh])

    # 保存
    o3d.io.write_triangle_mesh("delaunay_surface_mesh.ply", mesh)