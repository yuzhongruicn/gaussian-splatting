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
    parser = argparse.ArgumentParser(description='transfer IDG dataset to COLMAP format')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-mini', 'v1,0-trainval'])
    parser.add_argument('-s', '--data_path', type=str, default = '/root/paddlejob/workspace/yuzhongrui/datasets')
    parser.add_argument('--output_path', type=str)

    return parser.parse_args()


if __name__ == "__main__":

    args = read_args()

    dataroot = os.path.join(args.data_path, 'nuScenes', args.version.split('-')[-1])
    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=True)
    output_path = args.output_path

    scene_name = os.path.basename(output_path)

    scene_dict_path = os.path.join(dataroot, 'scene_dict.json')
    with open(scene_dict_path, 'r') as f:
        scene_dict = json.load(f)

    scene_token = scene_dict[scene_name]
    
    cam_to_worlds = {}
    image_names = []
    cam_intrinsics = {}

    scene = nusc.get('scene', scene_token)
    assert(scene['name'] == scene_name)

    IDX = 0
    IMAGE_NUM = 0
    sample_token = scene['first_sample_token']

    while sample_token:
        sample = nusc.get('sample', sample_token)

        for cam in sample['data']:
            if cam in CAMERAS:
                image_names.append(f'{scene_name}_{cam}_{IDX:04d}.png')
                cam_data = nusc.get('sample_data', sample['data'][cam])
            
                cam_intrinsic, cam2global = get_camera_params(nusc, cam_data)
                cam_to_worlds[f'{scene_name}_{cam}_{IDX:04d}.png'] = cam2global
                cam_intrinsics[f'{scene_name}_{cam}_{IDX:04d}.png'] = cam_intrinsic
                IMAGE_NUM += 1
        IDX += 1
        sample_token = sample['next']
        
    f_w = open(os.path.join(output_path, 'sparse/1/images.txt'), 'w')
    c_w = open(os.path.join(output_path, 'sparse/1/cameras.txt'), 'w')


    with open(os.path.join(output_path, 'image_list.txt'), 'r') as l_r:
        lines = l_r.readlines()
        print(lines)
        for line in lines:
            image_id, image_name, camera_id = line.strip().split()
            transform = cam_to_worlds[image_name]
            transform = np.linalg.inv(transform)
            r = R.from_matrix(transform[:3,:3])
            rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
            rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
            out = np.concatenate((rquat, transform[:3, 3]), axis=0)
            f_w.write(f'{image_id} ')
            f_w.write(' '.join([str(a) for a in out.tolist()] ) )
            f_w.write(f' {camera_id} {image_name}')
            f_w.write('\n\n')

            cam_intrinsic = cam_intrinsics[image_name]
            cx = cam_intrinsic[0,2]+0.5
            cy = cam_intrinsic[1,2]+0.5
            fx = cam_intrinsic[0,0]
            fy = cam_intrinsic[1,1]
            c_w.write(f'{image_id} PINHOLE 1600 900 {fx} {fy} {cx} {cy}')
            c_w.write('\n')
        

    f_w.close()
    l_r.close()
    c_w.close()




