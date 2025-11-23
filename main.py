from tools.mesh_culling import mesh_culling,file_2_mesh,file_2_pcd
import argparse
import os
from evaluation.Chamfer import chamfer_distance,compute_mesh_scale,chamfer_distance_points,compute_points_scale
from evaluation.F1 import f1_score,f1_score_points
from evaluation.normal_consistency import normal_consistency,normal_consistency_points
from evaluation.rendering import rendering_quality

os.environ["O3D_HEADLESS_RENDERING"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to COLMAP sparse model directory (e.g. sparse/0)")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to input mesh (e.g. model/mesh.ply)")
    parser.add_argument("--gt_mesh_path", type=str, required=True, help="Path to groundtruth mesh (e.g. model/mesh.ply)")
    args = parser.parse_args()
    #mesh_test = mesh_culling(args.model_dir,args.mesh_path)
    mesh_test = file_2_mesh(args.mesh_path)
    mesh_gt = file_2_pcd(args.gt_mesh_path)
    scale = compute_points_scale(mesh_gt)
    #print("Normalized Chamder Distance: ",chamfer_distance(mesh_test,mesh_gt,scale=scale))
    precision, recall, f1 = f1_score_points(mesh_test, mesh_gt, scale=scale)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    #f1, consistency = normal_consistency(mesh_test, mesh_gt)
    #print(f"Normal F1 Score: {f1}, Normal mean angle: {consistency}°")
    #psnr,ssim,lpips = rendering_quality(mesh_test,args.model_dir)
    #print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}")

if __name__ == "__main__":
    main()