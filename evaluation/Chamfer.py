import open3d as o3d
from scipy.spatial import cKDTree
import numpy as np

def sample_points_from_mesh(mesh, n_samples=100000):
    pcd = mesh.sample_points_uniformly(number_of_points=n_samples)
    return np.asarray(pcd.points)

def compute_mesh_scale(gt_mesh):
    vertices = np.asarray(gt_mesh.vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    scale = np.linalg.norm(max_bound - min_bound)  # 对角线长度
    return scale

def chamfer_distance(mesh_a, mesh_b, n_samples=100000, scale=1.0):
    print("Sampling points!")
    pts_a = sample_points_from_mesh(mesh_a, n_samples)
    pts_b = sample_points_from_mesh(mesh_b, n_samples)

    # 建立KD树加速最近邻查找
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    print("Calculating distances")
    # 单向Hausdorff距离
    dists_ab, _ = tree_b.query(pts_a, k=1)
    dists_ba, _ = tree_a.query(pts_b, k=1)

    cd_ab = np.mean(dists_ab ** 2)
    cd_ba = np.mean(dists_ba ** 2)
    chamfer = cd_ab + cd_ba
    chamfer /= scale ** 2
    print("Finished calculating!")

    return chamfer