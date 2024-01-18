import os
import sys
import numpy as np
import cv2
import json
import argparse
import open3d as o3d
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from utils.kitti_utils import load_intrinsics, load_cam_to_pose

def read_args():
    parser = argparse.ArgumentParser(description='transfer kitti360 dataset to COLMAP format')
    parser.add_argument('-s', '--data_path', type=str, default = '/root/paddlejob/workspace/yuzhongrui/datasets/kitti360' )
    parser.add_argument('-o', '--output_path', type=str, default='/root/paddlejob/workspace/yuzhongrui/datasets/kitti360/colmap_short')
    parser.add_argument('--drive', type=str, default='2013_05_28_drive_0009_sync')
    parser.add_argument('--skip', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = read_args()
    dataroot = os.path.join(args.data_path, "KITTI-360")
    output_path = os.path.join(args.output_path, args.drive)
    image_path = os.path.join(dataroot, 'data_2d_raw', args.drive, 'image_00/data_rect')
    pcd_path = os.path.join(dataroot, 'data_3d_semantics/train', args.drive, 'static')
    pcd_files = os.listdir(pcd_path)
    pcd_files.sort()
    
    pose_fn = os.path.join(dataroot, 'data_poses', args.drive, 'poses.txt')
    intrinsic_fn = os.path.join(dataroot, 'calibration', 'perspective.txt')
    cam2lidar_fn = os.path.join(dataroot, 'calibration', 'calib_cam_to_velo.txt') 
    cam2pose_fn = os.path.join(dataroot, 'calibration', 'calib_cam_to_pose.txt')

    P_rect_00, P_rect_01, R_rect_00_, R_rect_01_ = load_intrinsics(intrinsic_fn)
    R_rect_00 = np.eye(4)
    R_rect_00[:3, :3] = R_rect_00_
    R_rect_01 = np.eye(4)
    R_rect_01[:3, :3] = R_rect_01_

    c2p_00, _ = load_cam_to_pose(cam2pose_fn)
    c2p_00 = np.concatenate([c2p_00, np.array([[0, 0, 0, 1]])], axis=0)

    # poses
    poses = np.loadtxt(pose_fn)
    img_id = poses[:, 0].astype(np.int32)
    poses = poses[:, 1:].reshape(-1, 3, 4)
    pose_dict = {}
    poses = np.loadtxt(pose_fn)
    img_id = poses[:, 0].astype(np.int32)
    poses = poses[:, 1:].reshape(-1, 3, 4)
    pose_dict = {}
    for i in range(len(img_id)):
        img_name = f'{img_id[i]:010d}.png'
        pose_dict[img_name] = poses[i]
        
    # BLOCK_IDX=0
    pcd_files_chosen = ["0000000778_0000001026.ply", "0000001005_0000001244.ply", "0000002117_0000002353.ply", "0000002826_0000003034.ply"]
    range_list = [[880,960],[1102,1182],[2170,2250],[2900,2980]]

    for pcd_id, pcd_file in enumerate(pcd_files):
        if pcd_file not in pcd_files_chosen:
            continue
        print(f'processing {pcd_file}, block_{pcd_id}')
        os.makedirs(os.path.join(output_path, f'block_{pcd_id}', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, f'block_{pcd_id}', 'sparse', '0'), exist_ok=True)
        
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, pcd_file))
        o3d.io.write_point_cloud(os.path.join(output_path, f'block_{pcd_id}', 'sparse/0', 'points3D_all.ply'), pcd) 
        
        range_id = pcd_files_chosen.index(pcd_file)
        start_frame = range_list[range_id][0]
        end_frame = range_list[range_id][1]
        # start_frame = int(pcd_file.split('_')[0])
        # end_frame = int(pcd_file.split('.')[0].split('_')[-1])
        
        IMAGE_NUM = 0
        image_names = []
        cam_to_world = {}
        cam_intrinsic = {}
        
        for idx in range(start_frame, end_frame + 1):
            image_name = f'{idx:010d}.png'
            if not image_name in pose_dict.keys():
                continue
            image_names.append(image_name)
        
        image_names = image_names[args.skip:]
        
        f_w = open(os.path.join(output_path, f'block_{pcd_id}', 'sparse/0/images.txt'), 'w')
        c_w = open(os.path.join(output_path, f'block_{pcd_id}', 'sparse/0/cameras.txt'), 'w')
        
        for idx, image_name in enumerate(image_names): 
            
            img = cv2.imread(os.path.join(image_path, image_name))
            cv2.imwrite(os.path.join(output_path, f'block_{pcd_id}', 'images', f'{image_name}'), img)
            H, W = img.shape[:2]
            
            pose = pose_dict[image_name]
            c2w_00 = np.eye(4)
            c2w_00_ = np.matmul(np.matmul(pose, c2p_00), np.linalg.inv(R_rect_00))
            c2w_00[:c2w_00_.shape[0], :c2w_00_.shape[1]] = c2w_00_
            cam_to_world[image_name] = c2w_00
            
            # cam_intrinsic[image_name] = [P_rect_00, H, W]å
            # transform = cam_to_world[image_name]
            transform = np.linalg.inv(c2w_00)

            r = R.from_matrix(transform[:3,:3])
            rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
            rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
            out = np.concatenate((rquat, transform[:3, 3]), axis=0)
            f_w.write(f'{idx + 1} ')
            f_w.write(' '.join([str(a) for a in out.tolist()] ) )
            f_w.write(f' {idx + 1} {image_name}')
            f_w.write('\n\n')
            
            c_w.write(f'{idx + 1} PINHOLE {W} {H} {P_rect_00[0, 0]} {P_rect_00[1, 1]} {P_rect_00[0, 2]} {P_rect_00[1, 2]}')
            c_w.write('\n')

        f_w.close()
        c_w.close()
        print(f'finished processing {pcd_file}, block_{pcd_id}, total {len(image_names)} images')
        # BLOCK_IDX += 1
        # exit()








    





    

