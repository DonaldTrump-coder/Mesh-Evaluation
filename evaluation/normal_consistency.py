import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

def normal_consistency(mesh_a, mesh_b, n_samples=500000, angle_threshold=10.0):
    mesh_a.compute_vertex_normals()
    mesh_b.compute_vertex_normals()

    vertices_a = np.asarray(mesh_a.vertices)
    normals_a = np.asarray(mesh_a.vertex_normals)

    vertices_b = np.asarray(mesh_b.vertices)
    normals_b = np.asarray(mesh_b.vertex_normals)

    print("Sampling points from meshes...")
    pcd_a = mesh_a.sample_points_uniformly(number_of_points=n_samples)
    pts_a = np.asarray(pcd_a.points)
    tree_vertices_a = cKDTree(vertices_a)
    _, idx_a = tree_vertices_a.query(pts_a)
    sampled_normals_a = normals_a[idx_a]

    pcd_b = mesh_b.sample_points_uniformly(number_of_points=n_samples)
    pts_b = np.asarray(pcd_b.points)
    tree_vertices_b = cKDTree(vertices_b)
    _, idx_b = tree_vertices_b.query(pts_b)
    sampled_normals_b = normals_b[idx_b]
    tree_b_points = cKDTree(pts_b)
    tree_a_points = cKDTree(pts_a)

    # pts_a -> pts_b
    _, idx_a2b = tree_b_points.query(pts_a)
    corresponding_normals_b = sampled_normals_b[idx_a2b]

    # pts_b -> pts_a
    _, idx_b2a = tree_a_points.query(pts_b)
    corresponding_normals_a = sampled_normals_a[idx_b2a]

    def angle_between_normals(n1, n2):
        dot = np.einsum('ij,ij->i', n1, n2)
        dot = np.clip(np.abs(dot), 0.0, 1.0)
        return np.degrees(np.arccos(dot))

    angles_a = angle_between_normals(sampled_normals_a, corresponding_normals_b)
    angles_b = angle_between_normals(sampled_normals_b, corresponding_normals_a)

    # True positives
    tp_a = np.sum(angles_a <= angle_threshold)
    tp_b = np.sum(angles_b <= angle_threshold)

    precision = tp_a / len(angles_a) if len(angles_a) > 0 else 0.0
    recall = tp_b / len(angles_b) if len(angles_b) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    mean_angle = np.mean(np.concatenate([angles_a, angles_b]))
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    print("Finished calculating normal consistency!")
    return f1, mean_angle

def normal_consistency_points(mesh_a, points_b, n_samples=1000000, angle_threshold=5.0):
    mesh_a.compute_vertex_normals()
    vertices_a = np.asarray(mesh_a.vertices)
    normals_a = np.asarray(mesh_a.vertex_normals)

    pcd_a = mesh_a.sample_points_uniformly(number_of_points=n_samples)
    pts_a = np.asarray(pcd_a.points)

    tree_vertices_a = cKDTree(vertices_a)
    _, idx_a = tree_vertices_a.query(pts_a)
    sampled_normals_a = normals_a[idx_a]

    if not points_b.has_normals():
        points_b.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        points_b.orient_normals_consistent_tangent_plane(k=6)

    vertices_b = np.asarray(points_b.points)
    normals_b = np.asarray(points_b.normals)

    num_points_b = vertices_b.shape[0]
    if num_points_b > n_samples:
        idx = np.random.choice(num_points_b, n_samples, replace=False)
        vertices_b = vertices_b[idx]
        normals_b = normals_b[idx]

    pts_b = vertices_b
    sampled_normals_b = normals_b

    # --- KDTree ---
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    _, idx_a2b = tree_b.query(pts_a)
    _, idx_b2a = tree_a.query(pts_b)

    corresponding_normals_b = sampled_normals_b[idx_a2b]
    corresponding_normals_a = sampled_normals_a[idx_b2a]

    def angle_between_normals(n1, n2):
        dot = np.einsum('ij,ij->i', n1, n2)
        dot = np.clip(np.abs(dot), 0.0, 1.0)
        return np.degrees(np.arccos(dot))

    angles_a = angle_between_normals(sampled_normals_a, corresponding_normals_b)
    angles_b = angle_between_normals(sampled_normals_b, corresponding_normals_a)

    # True positives & F1
    tp_a = np.sum(angles_a <= angle_threshold)
    tp_b = np.sum(angles_b <= angle_threshold)
    precision = tp_a / len(angles_a)
    recall = tp_b / len(angles_b)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    mean_angle = np.mean(np.concatenate([angles_a, angles_b]))

    print("Finished calculating normal consistency!")
    return f1, mean_angle