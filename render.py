import argparse
import numpy as np
import open3d as o3d
import cv2
import os
import struct
from tqdm import tqdm
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None
def read_cameras_bin(path):
    cams = {}
    with open(path, 'rb') as f:
        num_cams = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cams):
            cam_id = struct.unpack('<i', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            w = int(struct.unpack('<Q', f.read(8))[0])
            h = int(struct.unpack('<Q', f.read(8))[0])
            if model_id == 1:
                fx = fy = struct.unpack('<d', f.read(8))[0]
                cx, cy = struct.unpack('<d', f.read(8))[0], struct.unpack('<d', f.read(8))[0]
            elif model_id == 0:
                fx, fy = struct.unpack('<d', f.read(8))[0], struct.unpack('<d', f.read(8))[0]
                cx, cy = struct.unpack('<d', f.read(8))[0], struct.unpack('<d', f.read(8))[0]
            else:
                raise ValueError(f"Unknown model_id: {model_id}")
            cams[cam_id] = (w, h, fx, fy, cx, cy)
    return cams
def read_images_bin(path):
    imgs = []
    with open(path, 'rb') as f:
        num_imgs = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_imgs):
            img_id = struct.unpack('<i', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
            tx, ty, tz = struct.unpack('<ddd', f.read(24))
            cam_id = struct.unpack('<i', f.read(4))[0]
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch
            filename = name.decode('utf-8')
            n_points = struct.unpack('<Q', f.read(8))[0]
            f.seek(24 * n_points, 1)
            imgs.append((cam_id, qw, qx, qy, qz, tx, ty, tz, filename))
    return imgs
def quat_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,   1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx**2-2*qy**2]
    ])
def get_camera_params(sparse_dir):
    cameras_data = read_cameras_bin(os.path.join(sparse_dir, 'cameras.bin'))
    images_data = read_images_bin(os.path.join(sparse_dir, 'images.bin'))
    root_dir = os.path.dirname(os.path.dirname(sparse_dir))
    image_dir = os.path.join(root_dir, 'images')
    cam_list = []
    for cam_id, qw, qx, qy, qz, tx, ty, tz, fname in images_data:
        w, h, fx, fy, cx, cy = cameras_data[cam_id]
        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        img_path = os.path.join(image_dir, os.path.basename(fname))
        cam_list.append({
            'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            'R': R, 't': t,
            'width': w, 'height': h,
            'image_path': img_path
        })
    return cam_list
def build_scene(mesh_legacy):
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    return scene
def render_ray(scene, mesh_legacy, K, R_w2c, t_w2c, width, height):
    C = -R_w2c.T @ t_w2c
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_w2c
    extrinsic[:3, 3] = C
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=K, extrinsic_matrix=extrinsic,
        width_px=width, height_px=height
    )
    ans = scene.cast_rays(rays)
    hit = ans['t_hit'].isfinite()
    mask = hit.numpy()
    if mask.sum() < 100:
        return np.full((height, width, 3), 255, dtype=np.uint8), mask
    vc = np.asarray(mesh_legacy.vertex_colors)
    if vc.shape[-1] == 4:
        vc = vc[:, :3]
    faces = np.asarray(mesh_legacy.triangles)
    prim_ids = ans['primitive_ids'][hit].numpy().astype(np.int64)
    uv = ans['primitive_uvs'][hit].numpy()
    u, v = uv[:, 0], uv[:, 1]
    w_bary = 1.0 - u - v
    tri = faces[prim_ids]
    c0, c1, c2 = vc[tri[:, 0]], vc[tri[:, 1]], vc[tri[:, 2]]
    color_hit = np.clip(w_bary[:, None] * c0 + u[:, None] * c1 + v[:, None] * c2, 0, 1)
    img = np.full((height, width, 3), 1.0, dtype=np.float64)
    yy, xx = np.where(mask)
    img[yy, xx] = color_hit
    return (img * 255).astype(np.uint8), mask
def evaluate_mesh_rendering(mesh_path, sparse_dir, max_images=None):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        vn = np.asarray(mesh.vertex_normals)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip((vn + 1) / 2, 0, 1))
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
    scene = build_scene(mesh)
    cameras = get_camera_params(sparse_dir)
    print(f"Cameras: {len(cameras)}")
    if max_images:
        cameras = cameras[:max_images]
    psnr_list, ssim_list, skipped = [], [], 0
    for cam in tqdm(cameras, desc="Rendering"):
        if not os.path.exists(cam['image_path']):
            skipped += 1
            continue
        rendered, mask = render_ray(scene, mesh,
                                     cam['K'], cam['R'], cam['t'],
                                     cam['width'], cam['height'])
        real = cv2.imread(cam['image_path'])
        if real is None:
            skipped += 1
            continue
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        if rendered.shape[:2] != real.shape[:2]:
            rendered = cv2.resize(rendered, (real.shape[1], real.shape[0]))
            mask = mask & ~(rendered == 255).all(axis=2)
        if mask.sum() < 100:
            skipped += 1
            continue
        diff = rendered[mask].astype(float) - real[mask].astype(float)
        mse = np.mean(diff ** 2)
        psnr = 100.0 if mse < 1e-10 else 10 * np.log10(255.0 ** 2 / mse)
        psnr_list.append(psnr)
        if ssim is not None:
            ssim_list.append(ssim(rendered, real, channel_axis=2, data_range=255))
    print(f"\n===== Rendering Evaluation (Open3D Raycasting) =====")
    print(f"Valid: {len(psnr_list)}/{len(cameras)}, skipped: {skipped}")
    if psnr_list:
        print(f"PSNR:  {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f} dB")
    if ssim_list:
        print(f"SSIM:  {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    return np.mean(psnr_list) if psnr_list else 0.0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()
    evaluate_mesh_rendering(args.mesh_path, args.model_dir, args.max_images)