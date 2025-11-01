from tools.mesh_culling import mesh_culling,file_2_mesh
import argparse
import os
from evaluation.Chamfer import chamfer_distance,compute_mesh_scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to COLMAP sparse model directory (e.g. sparse/0)")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to input mesh (e.g. model/mesh.ply)")
    parser.add_argument("--gt_mesh_path", type=str, required=True, help="Path to groundtruth mesh (e.g. model/mesh.ply)")
    args = parser.parse_args()
    mesh_test = mesh_culling(args.model_dir,args.mesh_path)
    mesh_gt = file_2_mesh(args.gt_mesh_path)
    scale = compute_mesh_scale(mesh_gt)
    print(chamfer_distance(mesh_test,mesh_gt,scale=scale))

if __name__ == "__main__":
    main()