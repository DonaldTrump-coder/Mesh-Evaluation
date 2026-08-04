import open3d as o3d
import numpy as np
import tools.colmap as colmap
from scipy.spatial.transform import Rotation as R
import tqdm
import os
import cv2
from tools.image_metrics import compute_psnr,compute_ssim,compute_lpips
import torch
import trimesh
import pyrender

def fix_colmap_extrinsic_for_open3d(extrinsic):
    extrinsic_fixed = extrinsic.copy()
    
    R_flip = np.diag([1, -1, -1])
    
    extrinsic_fixed[:3, :3] = R_flip @ extrinsic[:3, :3]
    return extrinsic_fixed

def colmap_camera_to_pyrender(cameras, images, image_id):
    image = images[image_id]
    camera = cameras[image.camera_id]

    params = camera.params
    if camera.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")

    width = camera.width
    height = camera.height

    intrinsic = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height
    }

    qvec = image.qvec
    tvec = image.tvec.reshape(3,1)

    # COLMAP quaternion -> rotation matrix
    R_wc = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

    # Camera-to-world
    R_cw = R_wc.T
    t_cw = -R_cw @ tvec

    R_flip = np.diag([1, 1, -1])
    t_flip = np.zeros(3)

    R_final = R_flip @ R_cw
    t_final = R_flip @ t_cw.flatten()

    pose = np.eye(4)
    pose[:3, :3] = R_final
    pose[:3, 3] = t_final

    return intrinsic, pose

def rendering_quality(mesh, model_dir: str):
    image_dir = os.path.join(os.path.dirname(os.path.dirname(model_dir)), "images")
    cameras, images, points3D = colmap.read_model(model_dir, ext=".bin")
    width, height = next(iter(cameras.values())).width, next(iter(cameras.values())).height

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(vertex_colors*255).astype(np.uint8))

    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    rendered_images = {}

    for image_id, image in tqdm.tqdm(images.items(), desc="Rendering"):
        intrinsic, pose = colmap_camera_to_pyrender(cameras, images, image_id)

        camera = pyrender.IntrinsicsCamera(
            fx=intrinsic["fx"], fy=intrinsic["fy"],
            cx=intrinsic["cx"], cy=intrinsic["cy"],
            znear=0.001, zfar=1000.0
        )

        scene = pyrender.Scene(bg_color=[1.0,1.0,1.0,1.0], ambient_light=[1.0,1.0,1.0])
        scene.add(pyrender_mesh)
        scene.add(camera, pose=pose)

        img_np, _ = r.render(scene)
        rendered_images[image.name] = img_np
        print(image.name)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0,0,0

    psnr_list = []
    ssim_list = []
    lpips_list = []

    for name, render_np in tqdm.tqdm(rendered_images.items(), desc="Evaluating"):
        img_path = os.path.join(image_dir, name)

        gt_np = cv2.imread(img_path)
        gt_np = cv2.cvtColor(gt_np, cv2.COLOR_BGR2RGB)

        psnr_val = compute_psnr(render_np, gt_np)
        psnr_list.append(psnr_val)

        ssim_val = compute_ssim(render_np, gt_np)
        ssim_list.append(ssim_val)

        render_tensor = torch.from_numpy(render_np.astype(np.float32)/127.5 - 1.0).permute(2,0,1).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_np.astype(np.float32)/127.5 - 1.0).permute(2,0,1).unsqueeze(0)

        lpips_val = compute_lpips(render_tensor, gt_tensor, net='alex', use_gpu=True)
        lpips_list.append(lpips_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    return avg_psnr, avg_ssim, avg_lpips