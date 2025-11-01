import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def sample_points_from_mesh(mesh, n_samples=100000):
    pcd = mesh.sample_points_uniformly(number_of_points=n_samples)
    return np.asarray(pcd.points)

def f1_score(mesh_a, mesh_b, n_samples=100000, threshold=0.01, scale=1.0):
    print("Sampling points from meshes...")
    pts_a = sample_points_from_mesh(mesh_a, n_samples)
    pts_b = sample_points_from_mesh(mesh_b, n_samples)

    print("Building KD-Trees...")
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    # 归一化距离阈值
    norm_threshold = threshold * scale
    # pts_a 到 pts_b 的距离
    dists_a, _ = tree_b.query(pts_a, k=1)
    # pts_b 到 pts_a 的距离
    dists_b, _ = tree_a.query(pts_b, k=1)

    # True positives
    tp_a = np.sum(dists_a <= norm_threshold)
    tp_b = np.sum(dists_b <= norm_threshold)

    precision = tp_a / len(pts_a) if len(pts_a) > 0 else 0
    recall = tp_b / len(pts_b) if len(pts_b) > 0 else 0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print("Finished Calculating!")
    return precision, recall, f1