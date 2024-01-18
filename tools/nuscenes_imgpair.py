'''
partially from S-NeRF
'''
import numpy as np
import os
import cv2
import json
from PIL import Image

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from collections import defaultdict
import argparse
from utils.nuscenes_utils import mask_outside_points
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes

from copy import deepcopy
import random

import open3d as o3d


CAMERAS = [
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT'
]

def get_camera_params(nusc, cam_data):
    # intrinsic
    cam_cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(cam_cs_rec['camera_intrinsic'])

    # vehicle to cam
    cam2vehicle_rotation = cam_cs_rec['rotation']
    cam2vehicle_translation = cam_cs_rec['translation']
    cam2vehicle = transform_matrix(cam2vehicle_translation, Quaternion(cam2vehicle_rotation))

    # vehicle to global at camera timestep
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    vehicle2global_rotation = pose_record['rotation']
    vehicle2global_translation = pose_record['translation']
    vehicle2global = transform_matrix(vehicle2global_translation, Quaternion(vehicle2global_rotation))

    cam2global = vehicle2global @ cam2vehicle

    return cam_intrinsic, cam2global
    
def get_lidar_params(nusc, lidar_data):

    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar2vehicle_rotation = cs_record['rotation']
    lidar2vehicle_translation = cs_record['translation']
    lidar2vehicle = transform_matrix(lidar2vehicle_translation, Quaternion(lidar2vehicle_rotation))
    
    # vehicle to global
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    vehicle2global_rotation = pose_record['rotation']
    vehicle2global_translation = pose_record['translation']
    vehicle2global = transform_matrix(vehicle2global_translation, Quaternion(vehicle2global_rotation))
    
    # lidar to global
    lidar2global = vehicle2global @ lidar2vehicle

    return lidar2global

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-mini', 'v1.0-trainval'])
    parser.add_argument('--datadir', type=str, default = './data' )
    parser.add_argument('--skip', type=int, default=20, help = 'Skip the first N frames') # Caution, this may vary from scene to scene
    parser.add_argument('--total_num', type=int, default = 40, help = 'The frames needed')
    parser.add_argument('--camera_index', type=list, default = [0,1,2,3,4,5], help = 'Cameras chosen')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--savedir', type=str, default='./data/scenes')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=900)
    args = parser.parse_args()

    # Data Initialization
    dataroot = os.path.join(args.datadir, 'nuScenes', args.version.split('-')[-1])
    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=True)
    scene_dict_path = os.path.join(dataroot, 'scene_dict.json')
    # val_scene_dict_path = os.path.join(dataroot, 'scene_dict_val.json')

    save_dir = args.savedir

    prompt_json_path = os.path.join(save_dir, 'prompt.json')
    promt_json_list = []
    cameras = [CAMERAS[i] for i in args.camera_index]

    # splits = create_splits_scenes()
    # # train_split = 'train'
    # print(len(splits['train']))
    # print(len(splits['val']))
    # train_scene_dict = {}
    # val_scene_dict = {}
    # for scene in nusc.scene:
    #     if scene['name'] in splits['train']:
    #         train_scene_dict[scene['name']] = scene['token']
    #     elif scene['name'] in splits['val']:
    #         val_scene_dict[scene['name']] = scene['token']

    # with open(scene_dict_path, 'w') as f:
    #     json.dump(train_scene_dict, f)
    # with open(val_scene_dict_path, 'w') as f:
    #     json.dump(val_scene_dict, f)
    # exit()

    img_savepath = os.path.join(save_dir, 'images')
    # proj_savepath = os.path.join(save_dir, 'projcted')
    # sparse_savepath = os.path.join(proj_savepath, 'sparse')
    dense_savepath = os.path.join(save_dir, 'dense')
    pcd_savepath = os.path.join(save_dir, 'pcd')
    os.makedirs(os.path.join(img_savepath, 'train'), exist_ok=True)
    os.makedirs(os.path.join(img_savepath, 'test'), exist_ok=True)
    os.makedirs(os.path.join(dense_savepath, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dense_savepath, 'test'), exist_ok=True)
    os.makedirs(dense_savepath, exist_ok=True)
    os.makedirs(pcd_savepath, exist_ok=True)
    
    with open(scene_dict_path, 'r') as f:
        scene_dict = json.load(f)
    
    for scene_name, scene_token in scene_dict.items():
        images, ego2global_rts, cam2ego_rts, cam_intrinsics = [], [], [], []
        print(f"Start processing scene {scene_name} | token: {scene_token}")
        scene = nusc.get('scene', scene_token)
        assert(scene['name'] == scene_name), "Scene name not match"

        # save_dir = os.path.join(args.savedir, scene_name)

        sample_token = scene['first_sample_token']
            
        IDX = 0
        channel_tokens = defaultdict(list)

        all_points = np.empty((0, 4))
        all_colors = np.empty((0, 3))

        #get all lidar points
        while sample_token:
            
            sample = nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            pointcloud, _ = LidarPointCloud.from_file_multisweep(nusc, sample, chan='LIDAR_TOP', ref_chan='LIDAR_TOP', nsweeps=5)

            # lidar to vehicle
            lidar2global = get_lidar_params(nusc, lidar_data)
            pointcloud.transform(lidar2global)
    
            # colors = np.empty((0, 3), dtype=np.float32)
            colors = np.zeros((pointcloud.points.shape[1], 3), dtype=np.float32)
            point_camera_dist = np.ones((pointcloud.points.shape[1], 1), dtype=np.float32) * np.inf
            valid_mask_all = np.zeros((pointcloud.points.shape[1]), dtype=bool)

            for cam in sample['data']:
                if cam in cameras:
                    cam_data = nusc.get('sample_data', sample['data'][cam])
                    cam_img_path = nusc.get_sample_data_path(cam_data['token'])
                    img = cv2.imread(cam_img_path)

                    cam_intrinsic, cam2global = get_camera_params(nusc, cam_data)

                    global2cam = np.linalg.inv(cam2global)

                    pointcloud_cam = deepcopy(pointcloud)
                    pointcloud_cam.transform(global2cam)
                    depths = pointcloud_cam.points[2, :]
                    points = view_points(pointcloud_cam.points[:3, :], cam_intrinsic, normalize=True)

                    points_frame, depths_frame, valid_mask = mask_outside_points(points, depths, height=900)
                    colors_frame = colors[valid_mask]
                    point_cam_dist_frame = point_camera_dist[valid_mask]
                    
                    # points_img = points2im(points_frame, img)
                    # cv2.imwrite(os.path.join(sparse_savepath, f'{cam}_{IDX:04d}_sparse.png'), points_img)
                    # Image.fromarray(points_img).save(os.path.join(proj_savepath, f'{cam}_{IDX:04d}_proj.png'))

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

            IDX += 1

            all_points = np.vstack((all_points, pointcloud.points.T[valid_mask_all]))
            all_colors = np.vstack((all_colors, colors[valid_mask_all]))
            sample_token = sample['next']
        
        print(f"Finish processing lidar pcd of scene {scene_name}, total {IDX} frames")
        pointcloud_all = LidarPointCloud(all_points.T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points[:, :3])  # xyz
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.io.write_point_cloud(os.path.join(pcd_savepath, f"{scene_name}_all_points.ply"), pcd)
        
    # exit()


        first_sample = nusc.get('sample', scene['first_sample_token'])
        for i in range(args.skip):
            first_sample = nusc.get('sample', first_sample['next'])
        
        last_sample = nusc.get('sample', scene['last_sample_token'])
        for i in range(args.skip):
            last_sample = nusc.get('sample', last_sample['prev'])
        last_sample_token = last_sample['token']


        IDX = 0
        NUMS = 0
        
        # #project all pcd to images
        sample_token = first_sample['token']

        while sample_token and sample_token != last_sample_token:
            sample = nusc.get('sample', sample_token)
            for cam in sample['data']:
                if cam in cameras:
                    NUMS += 1
                    promt_dict = {}
                    cam_data = nusc.get('sample_data', sample['data'][cam])
                    cam_img_path = nusc.get_sample_data_path(cam_data['token'])
                    img = cv2.imread(cam_img_path)
                    
                    proj_img = np.zeros_like(img)
                    z_img = np.ones((img.shape[0], img.shape[1])) * np.inf

                    cam_intrinsic, cam2global = get_camera_params(nusc, cam_data)
                    global2cam = np.linalg.inv(cam2global)

                    pointcloud_cam = deepcopy(pointcloud_all)
                    pointcloud_cam.transform(global2cam)
                    depths = pointcloud_cam.points[2, :]
                    points = view_points(pointcloud_cam.points[:3, :], cam_intrinsic, normalize=True)
                    points_frame, depths_frame, valid_mask = mask_outside_points(points, depths, height=900)
                    
                    # colors_frame = all_colors[valid_mask]
                    
                    for i, point in enumerate(points_frame.T):
                        x, y = int(point[0]), int(point[1])
                        z = depths_frame[i]
                        if z_img[y, x] > z:
                            proj_img[y, x, :] = img[y, x, :]
                            z_img[y, x] = z
                    
                    if (NUMS+1) % 100 == 0:
                        cv2.imwrite(os.path.join(img_savepath, 'test', f'{scene_name}_{cam}_{IDX:04d}.png'), img)
                        cv2.imwrite(os.path.join(dense_savepath, 'test', f'{scene_name}_{cam}_{IDX:04d}.png'), proj_img)
                    else:
                        promt_dict["source"] = os.path.join(dense_savepath, 'train', f'{scene_name}_{cam}_{IDX:04d}.png')
                        promt_dict["target"] = os.path.join(img_savepath, 'train', f'{scene_name}_{cam}_{IDX:04d}.png')
                        promt_dict["prompt"] = "realistic streetview"
                        promt_json_list.append(promt_dict)
                        cv2.imwrite(os.path.join(img_savepath, 'train', f'{scene_name}_{cam}_{IDX:04d}.png'), img)
                        cv2.imwrite(os.path.join(dense_savepath, 'train', f'{scene_name}_{cam}_{IDX:04d}.png'), proj_img)
            IDX += 1
            sample_token = sample['next']
            
        print(f"Finish processing images in scene {scene_name}, total {IDX} frames, {NUMS} images")

    # random.shuffle(promt_json_list)
    with open(prompt_json_path, 'w') as f:
        for entry in promt_json_list:
            f.write(json.dumps(entry) + '\n')

    





        # for s in sensor:
        #     temp_data = nusc.get('sample_data', temp_sample['data'][s])
        #     for i in range(args.total_num):      
        #         data_path , _ , cam_intrinsic = nusc.get_sample_data(temp_data['token'])
        #         if not os.path.exists(data_path):
        #             import pdb; pdb.set_trace()
                
        #         channel_tokens[s].append(temp_data['token'])
        #         if(temp_data['is_key_frame']):
        #             sample_idx_list[IDX] = temp_data['token']
        #         IDX += 1

        #         cam_intrinsics.append(cam_intrinsic.astype(np.float32))
        #         #image
        #         fname = data_path
        #         img = cv2.imread(fname)
        #         cv2.imwrite(os.path.join(img_savepath, f'{s}_{i:04d}.png'), img)
        #         images.append(img)

        #         #ego2global
        #         temp_ego2global = nusc.get('ego_pose',temp_data['ego_pose_token'])
        #         ego2global_r = Quaternion(temp_ego2global['rotation']).rotation_matrix
        #         ego2global_t = np.array(temp_ego2global['translation'])
        #         ego2global_rt = np.eye(4)
        #         ## correct the ego2global pose
        #         ego2global_rt[:3,:3] = ego2global_r
        #         ego2global_rt[:3,3] = ego2global_t
                
        #         ego2global_rts.append(ego2global_rt.astype(np.float32))
        #         temp_cam2ego = nusc.get('calibrated_sensor',temp_data['calibrated_sensor_token'])
        #         #cam2ego
        #         cam2ego_r=Quaternion(temp_cam2ego['rotation']).rotation_matrix
        #         cam2ego_t=np.array(temp_cam2ego['translation'])
        #         cam2ego_rt = np.eye(4)
        #         cam2ego_rt[:3, :3] = cam2ego_r
        #         cam2ego_rt[:3, 3] = cam2ego_t
        #         cam2ego_rts.append(cam2ego_rt.astype(np.float32))

        #         #pcd
        #         if s in ['CAM_FRONT']:
        #             min_dist = 2.5
        #         elif s in ['CAM_BACK']:
        #             min_dist = 6
        #         else:
        #             min_dist = 1.5      
                
        #         if s in ['CAM_FRONT', 'CAM_BACK']:
        #             frames = 50
        #         else:
        #             frames = 6
        #         current_points, current_pc = map_pointcloud_to_image(nusc, temp_data['token'], 900, frames, s, min_dist)
        #         gt = points2im(current_points)        
        
            
        #         Image.fromarray(gt).save(os.path.join(proj_savepath, f'{s}_{i:04d}_proj.png'))
                
        #         if temp_data['next'] == '':
        #             print(f"Sensor {s} Done, Total {len(images)} images")
        #             break
        #         temp_data = nusc.get('sample_data', temp_data['next'])

        # camtoworlds = [ego2global_rts[i] @cam2ego_rts[i] for i in range(len(cam2ego_rts))]
        # camtoworlds = np.stack(camtoworlds,axis=0)
        
        # #* Save the data
        # save_dir = os.path.join(args.savedir, scene_name)
        # # Change the image array to img
        # print('the number of total images:', len(images))
        # path = os.path.join(save_dir, 'images')
        # os.makedirs(path,exist_ok=True)
        # # Resize the image to the destermined resolution
        
        # width = args.width
        # height = args.height
        
        # for i in range(len(images)):
        #     cv2.imwrite(path+'/{:04d}.png'.format(i),images[i])

        