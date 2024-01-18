import os
import sys
import numpy as np
import cv2
import json
import argparse
import open3d as o3d
from copy import deepcopy

from utils.kitti_utils import load_intrinsics, load_cam_to_pose, view_points, mask_outside_points
from utils.graphics_utils import rotationMatrixToEulerAngles, radian2angle, angle2radian, eulerAngles2rotationMat

def read_args():
    parser = argparse.ArgumentParser(description='transfer nuScenes dataset to COLMAP format')
    parser.add_argument('-s', '--data_path', type=str, default = '/root/paddlejob/workspace/yuzhongrui/datasets/kitti360' )
    parser.add_argument('-o', '--output_path', type=str, default='/root/paddlejob/workspace/yuzhongrui/datasets/kitti360/imgproj')
    # parser.add_argument('--drive', type=str, default='2013_05_28_drive_0000_sync')

    return parser.parse_args()


if __name__ == '__main__':
    args = read_args()
    dataroot = os.path.join(args.data_path, "KITTI-360")
    output_path = os.path.join(args.output_path)
    
    os.makedirs(output_path, exist_ok=True)
    
    scene_dict_path = os.path.join(dataroot, 'scene_dict.json')  
    with open(scene_dict_path, 'r') as f:
        scene_dict = json.load(f)
    
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
        
    
    for scene in scene_dict.keys():
        img_savepath = os.path.join(output_path, scene)
        os.makedirs(os.path.join(img_savepath), exist_ok=True)
        print(f'processing {scene}')
        IMAGE_NUM = 0
        pose_fn = os.path.join(dataroot, 'data_poses', scene, 'poses.txt')
        image_path = os.path.join(dataroot, 'data_2d_raw', scene, 'image_00/data_rect')
        pcd_path = os.path.join(dataroot, 'data_3d_semantics/train', scene, 'static')
        pcd_files = os.listdir(pcd_path)
        pcd_files.sort()

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

        thetas = [-90, -60, -45, -30, 0, 30, 45, 60, 90]
        for pcd_file in pcd_files[1:2]:
            print(f'processing {pcd_file}')
            pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, pcd_file))
            start_frame = int(pcd_file.split('_')[0])
            end_frame = int(pcd_file.split('.')[0].split('_')[-1])

            colors = np.array(pcd.colors)[:, ::-1]
            num_points = np.array(pcd.points).shape[0]

            for idx in range(start_frame, end_frame + 1):
                img_name = f'{idx:010d}.png'
                if not img_name in pose_dict.keys():
                    continue    
                img = cv2.imread(os.path.join(image_path, img_name))

                pose = pose_dict[img_name]
                c2w_00 = np.eye(4)
                c2w_00_ = np.matmul(np.matmul(pose, c2p_00), np.linalg.inv(R_rect_00))
                # c2w_00[:c2w_00_.shape[0], :c2w_00_.shape[1]] = c2w_00_
                euler = rotationMatrixToEulerAngles(c2w_00_[:3, :3])
                angle = radian2angle(euler)
                
                for theta in thetas:
                    proj_img = np.zeros_like(img)
                    z_img = np.ones((img.shape[0], img.shape[1])) * np.inf
                    angle_copy = deepcopy(angle)
                    angle_copy[2] = angle[2] + theta
                    
                    rot = eulerAngles2rotationMat(angle2radian(angle_copy))
                    
                    c2w_00[:3, :3] = rot
                    c2w_00[:3, 3] = c2w_00_[:, -1]

                    w2c_00 = np.linalg.inv(c2w_00)
                    pcd_cam = deepcopy(pcd).transform(w2c_00)

                    depths = np.array(pcd_cam.points)[:, 2]

                    points = view_points(np.array(pcd_cam.points).T[:3, :], P_rect_00, normalize=True)
                    points_frame, depths_frame, valid_mask = mask_outside_points(points, depths, height=376)
                    print(np.sum(valid_mask))

                    colors_frame = colors.T[:, valid_mask]

                    for i, point in enumerate(points_frame.T):
                        x, y = int(point[0]), int(point[1])
                        z = depths_frame[i]
                        if z_img[y, x] > z:
                            # proj_img[y, x, :] = img[y, x, :]
                            proj_img[y, x, :] = colors_frame[:, i] * 255
                            z_img[y, x] = z
                    cv2.imwrite(os.path.join(img_savepath, f'{theta}_{scene}_{img_name}'), proj_img)
                    IMAGE_NUM += 1
                    
        print(f"Finish processing images in scene {scene}, total {IMAGE_NUM} images")
        exit()
        

    










    





    

