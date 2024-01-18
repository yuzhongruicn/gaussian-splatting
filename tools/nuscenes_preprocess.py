'''
partially from S-NeRF
'''
import numpy as np
import os
import cv2
import json
from collections import defaultdict
import argparse
from copy import deepcopy
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from utils.nuscenes_utils import get_camera_params, get_lidar_params
from utils.nuscenes_utils import mask_outside_points

CAMERAS = [
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT'
]


def read_args():
    parser = argparse.ArgumentParser(description='transfer nuScenes dataset to COLMAP format')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-mini', 'v1,0-trainval'])
    parser.add_argument('-s', '--data_path', type=str, default = '/root/paddlejob/workspace/yuzhongrui/datasets' )
    parser.add_argument('-o', '--output_path', type=str, default='/root/paddlejob/workspace/yuzhongrui/datasets/nuScenes/colmap')
    parser.add_argument('--skip', type=int, default=0, help = 'Skip the first N frames') # Caution, this may vary from scene to scene
    parser.add_argument('--camera_index', type=list, default = [0,1,2,3,4,5], help = 'Cameras chosen')

    return parser.parse_args()


if __name__ == '__main__':
    args = read_args()

    # Data Initialization
    dataroot = os.path.join(args.data_path, 'nuScenes', args.version.split('-')[-1])
    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=True)
    save_dir = args.output_path

    cameras = [CAMERAS[i] for i in args.camera_index]
    scene_dict_path = os.path.join(dataroot, 'scene_dict.json')
    with open(scene_dict_path, 'r') as f:
        scene_dict = json.load(f)
        
    transform1 = np.array(
        [
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    transform2 = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    
    for scene_name, scene_token in scene_dict.items():
        # images, ego2global_rts, cam2ego_rts, cam_intrinsics = [], [], [], []
        print(f"Start processing scene {scene_name} | token: {scene_token}")
        scene = nusc.get('scene', scene_token)
        assert(scene['name'] == scene_name)

        sample_token = scene['first_sample_token']
        output_path = os.path.join(save_dir, scene_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, 'images'))
            os.makedirs(os.path.join(output_path, 'sparse', '0'))

        IDX = 0
        IMAGE_NUM = 0
        cam_to_worlds = {}
        cam_intrinsics = {}
        image_names = []
        all_points = np.empty((0, 4))
        all_colors = np.empty((0, 3))
        
        samples = [
            samp for samp in nusc.sample if nusc.get("scene", samp["scene_token"])["name"] == scene_name
        ]

        # sort by timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        # while sample_token:
        for sample in samples:
            
            # sample = nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            
            pcl_path = os.path.join(nusc.dataroot, lidar_data['filename'])
            pointcloud = LidarPointCloud.from_file(pcl_path)
            # pointcloud, _ = LidarPointCloud.from_file_multisweep(nusc, sample, chan='LIDAR_TOP', ref_chan='LIDAR_TOP', nsweeps=5)

            lidar2global = get_lidar_params(nusc, lidar_data)
            pointcloud.transform(lidar2global)

            colors = np.zeros((pointcloud.points.shape[1], 3), dtype=np.float32)
            point_camera_dist = np.ones((pointcloud.points.shape[1], 1), dtype=np.float32) * np.inf
            valid_mask_all = np.zeros((pointcloud.points.shape[1]), dtype=bool)

            for cam in sample['data']:
                if cam in cameras:
                    image_name = f'{scene_name}_{cam}_{IDX:04d}.png'
                    cam_data = nusc.get('sample_data', sample['data'][cam])
                    cam_img_path = nusc.get_sample_data_path(cam_data['token'])
                    img = cv2.imread(cam_img_path)
                    W, H = img.shape[1], img.shape[0]
                    cv2.imwrite(os.path.join(output_path, 'images', image_name), img)
                    image_names.append(image_name)

                    cam_intrinsic, cam2global = get_camera_params(nusc, cam_data)
                    # print(f'{cam}, {IDX}, {cam2global}, {np.linalg.inv(cam2global)}')
                  
                    cam_to_worlds[image_name] = cam2global
                    cam_intrinsics[image_name] = [cam_intrinsic, W, H]

                    global2cam = np.linalg.inv(cam2global)

                    pointcloud_cam = deepcopy(pointcloud)
                    pointcloud_cam.transform(global2cam)
                    depths = pointcloud_cam.points[2, :]
                    points = view_points(pointcloud_cam.points[:3, :], cam_intrinsic, normalize=True)

                    points_frame, depths_frame, valid_mask = mask_outside_points(points, depths, height=900)
                    colors_frame = colors[valid_mask]
                    point_cam_dist_frame = point_camera_dist[valid_mask]
                    
                    for i, point in enumerate(points_frame.T):
                        x, y = int(point[0]), int(point[1])
                        z = depths_frame[i]
                        if point_cam_dist_frame[i] > z:
                            color = img[y, x, :]
                            colors_frame[i] = color[::-1]/255
                            point_cam_dist_frame[i] = z
                # exit()
                    colors[valid_mask] = colors_frame
                    point_camera_dist[valid_mask] = point_cam_dist_frame
                    valid_mask_all = np.logical_or(valid_mask_all, valid_mask)
                    IMAGE_NUM += 1

            IDX += 1
            all_points = np.vstack((all_points, pointcloud.points.T[valid_mask_all]))
            all_colors = np.vstack((all_colors, colors[valid_mask_all]))
            sample_token = sample['next']

        print(f"Finish processing lidar pcd of scene {scene_name}, total {IDX} frames, {IMAGE_NUM} images")
        pointcloud_all = LidarPointCloud(all_points.T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points[:, :3])  # xyz
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.io.write_point_cloud(os.path.join(output_path, 'sparse/0', f"{scene_name}_all_points.ply"), pcd)

        f_w = open(os.path.join(output_path, 'sparse/0/images.txt'), 'w')
        c_w = open(os.path.join(output_path, 'sparse/0/cameras.txt'), 'w')
        
        image_names.sort()
        for idx, image_name in enumerate(image_names):
            transform = cam_to_worlds[image_name]
            # print(transform)
            transform = np.linalg.inv(transform)
            # print(transform)
            r = R.from_matrix(transform[:3,:3])
            rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
            rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
            out = np.concatenate((rquat, transform[:3, 3]), axis=0)
            f_w.write(f'{idx + 1} ')
            f_w.write(' '.join([str(a) for a in out.tolist()] ) )
            f_w.write(f' {idx + 1} {image_name}')
            f_w.write('\n\n')

            cam_intrinsic, W, H = cam_intrinsics[image_name]
            cx = cam_intrinsic[0,2]
            cy = cam_intrinsic[1,2]
            fx = cam_intrinsic[0,0]
            fy = cam_intrinsic[1,1]
            c_w.write(f'{idx + 1} PINHOLE {W} {H} {fx} {fy} {cx} {cy}')
            c_w.write('\n')

        f_w.close()
        c_w.close()