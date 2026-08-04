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
    scale = np.linalg.norm(max_bound - min_bound)
    return scale

def compute_points_scale(gt_mesh):
    vertices = np.asarray(gt_mesh.points)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    scale = np.linalg.norm(max_bound - min_bound)
    return scale

def chamfer_distance(mesh_a, mesh_b, n_samples=300000, scale=1.0):
    print("Sampling points!")
    n_a = min(n_samples, len(mesh_a.triangles) * 100)
    n_b = min(n_samples, len(mesh_b.triangles) * 100)
    pcd_a = mesh_a.sample_points_uniformly(number_of_points=n_a, use_triangle_normal=True)
    pcd_b = mesh_b.sample_points_uniformly(number_of_points=n_b, use_triangle_normal=True)
    pts_a = np.asarray(pcd_a.points)
    pts_b = np.asarray(pcd_b.points)
    normal_a = np.asarray(pcd_a.normals)
    normal_b = np.asarray(pcd_b.normals)

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    print("Calculating distances")
    dists_ab, nn_ab = tree_b.query(pts_a, k=1)
    dists_ba, nn_ba = tree_a.query(pts_b, k=1)

    cd_ab = np.mean(dists_ab ** 2)
    cd_ba = np.mean(dists_ba ** 2)
    chamfer = cd_ab + cd_ba
    chamfer /= scale ** 2
    hausdorff = max(dists_ab.max(), dists_ba.max()) / scale
    print("Finished calculating!")
    
    dot_ab = np.abs(np.sum(normal_a * normal_b[nn_ab], axis=1)).clip(0, 1)
    dot_ba = np.abs(np.sum(normal_b * normal_a[nn_ba], axis=1)).clip(0, 1)
    all_dots = np.concatenate([dot_ab, dot_ba])
    normal_error = np.degrees(np.mean(np.arccos(all_dots)))

    return chamfer, hausdorff, normal_error

def chamfer_distance_points(mesh_a, points_b, n_samples=300000, scale=1.0):
    print("Sampling points from mesh!")
    n_a = min(n_samples, np.asarray(mesh_a.triangles).shape[0] * 100)
    pcd_a = mesh_a.sample_points_uniformly(number_of_points=n_a, use_triangle_normal=True)
    pts_a = np.asarray(pcd_a.points)
    normal_a = np.asarray(pcd_a.normals)
    pts_b = np.asarray(points_b.points)
    if points_b.has_normals():
        normal_b = np.asarray(points_b.normals)
    else:
        print("GT has no normals, estimating...")
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(pts_b)
        pcd_tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd_tmp.orient_normals_consistent_tangent_plane(k=30)
        normal_b = np.asarray(pcd_tmp.normals)
    num_b = pts_b.shape[0]
    if num_b > n_samples:
        idx = np.random.choice(num_b, n_samples, replace=False)
        pts_b = pts_b[idx]
        normal_b = normal_b[idx]
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    print("Calculating distances")
    dists_ab, nn_ab = tree_b.query(pts_a, k=1)
    dists_ba, nn_ba = tree_a.query(pts_b, k=1)
    # Chamfer Distance
    chamfer = (np.mean(dists_ab ** 2) + np.mean(dists_ba ** 2)) / (scale ** 2)
    # Hausdorff Distance
    hausdorff = max(dists_ab.max(), dists_ba.max()) / scale
    # Mean Normal Error
    dot_ab = np.abs(np.sum(normal_a * normal_b[nn_ab], axis=1)).clip(0, 1)
    dot_ba = np.abs(np.sum(normal_b * normal_a[nn_ba], axis=1)).clip(0, 1)
    all_dots = np.concatenate([dot_ab, dot_ba])
    normal_error = np.degrees(np.mean(np.arccos(all_dots)))
    print("Finished calculating!")
    return chamfer, hausdorff, normal_error