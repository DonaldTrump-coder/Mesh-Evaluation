import tools.colmap as colmap
import numpy as np
import open3d as o3d
import os
from tools.post_processing import post_process_mesh

def file_2_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    return mesh

def get_intrinsics(camera):
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx = fy = f
    else:
        raise NotImplementedError(f"不支持的相机模型: {camera.model}")
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    return K

def mesh_culling(model_dir:str, # sparse/0
                 mesh_path:str
                 ):
    cameras, images, points3D = colmap.read_model(model_dir, ext=".bin")
    cameras_list = {}
    for image_id, image in images.items():
        cam = cameras[image.camera_id]
        K = get_intrinsics(cam)
        R = image.qvec2rotmat()
        t = image.tvec.reshape(3, 1)
        P = K @ np.hstack((R, t))
        cameras_list[image.name] = {
        "K": K,
        "R": R,
        "t": t,
        "P": P,
        "width": cam.width,
        "height": cam.height
        }
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    visible_faces_global = set()

    # 初始化射线场景
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_t)

    for img_name, cam_data in cameras_list.items():
        print(f"Processing {img_name} ...")

        K = cam_data["K"]
        R = cam_data["R"]
        t = cam_data["t"]
        width = cam_data["width"]
        height = cam_data["height"]

        # 相机外参矩阵 (4x4)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3:] = t

        # 创建光线（pinhole 模型）
        intrinsics_tensor = o3d.core.Tensor(K, o3d.core.Dtype.Float32)
        extrinsics_tensor = o3d.core.Tensor(extrinsics, o3d.core.Dtype.Float32)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsics_tensor,
            extrinsic_matrix=extrinsics_tensor,
            width_px=width,
            height_px=height,
        )

        # 执行射线投射
        ans = scene.cast_rays(rays)

        primitive_ids = ans['primitive_ids'].numpy()
        geometry_ids = ans['geometry_ids'].numpy()

        # 只保留属于 mesh 的索引
        valid_faces = primitive_ids[(primitive_ids != 2**32 - 1) & (geometry_ids == mesh_id)]
        if len(valid_faces) == 0:
            print(f"No visible faces detected for {img_name}")
            continue

        visible_faces_global.update(np.unique(valid_faces))

    visible_faces_global = [int(f) for f in visible_faces_global if 0 <= f < len(mesh.triangles)]
    if len(visible_faces_global) == 0:
        print("No visible faces found. Check your camera poses or mesh.")
        return
    unique_vertices = np.unique(triangles[visible_faces_global]).tolist()
    final_mesh = mesh.select_by_index(unique_vertices, cleanup=True)
    final_mesh.compute_vertex_normals()
    print("Culling finished!")
    final_mesh = post_process_mesh(final_mesh, 50)

    return final_mesh