import open3d as o3d
import numpy as np

def Possion_resconstruction(means:np.ndarray, normals:np.ndarray):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(means)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("possion_reconstructed.ply", mesh)