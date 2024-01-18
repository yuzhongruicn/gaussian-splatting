import numpy as np
import os.path as osp
import cv2
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image

MAX_WIDTH= 1600
MAX_HEIGHT = 900

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

def transform_points(nusc, cam, pointsensor, pc):
    '''
    help:
        transform points to the target camera.
    args:
        nusc: An instance of class 'nusc'.
        cam: the record of camera. (in table 'sample_data')
        pointsensor: the record of points. (in table 'sample_data')
        pc: A LidarPointCloud instance.
    return:
        depths
    Note:
        Calling this function will change the pc.points directly.
    '''
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    
    return depths


def mask_outside_points(points, depths, height):
    '''
    help:
        mask the points outside the image and change the axies.
    args:
        points: An [2, k] array.
        depths: An [k] array.
        height: the height of image.
    return:
        points and depths after masking.
    '''
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < MAX_WIDTH - 1)
    mask = np.logical_and(mask, points[1, :] > MAX_HEIGHT - height + 1)
    mask = np.logical_and(mask, points[1, :] < MAX_HEIGHT - 1)
    points = points[:, mask]
    depths = depths[mask]
    
    # Change the axies.
    points[1, :] -= (MAX_HEIGHT - height)  
    
    return points, depths, mask

def accumulate_points(nusc, pointsensor, type):
    '''
    help:
        accumulate another point cloud file.
    args:
        nusc: An instance of class 'nusc'.
        pointsensor: the record of points. (in table 'sample_data')
        type: "next" or "prev".
    return:
        pointsensor_xxxx: The "next" or "prev" of the pointsensor.
        pc_xxxx: The "next" or "prev" pointcloud.
    '''
    assert type in ("next", "prev"), "Invalid type. Type should be \"next\" or \"prev\"."
    if type == "next": 
        if not pointsensor['next']: 
            # No next file, return with no change.
            print("Warning: There is no more next pointcloud file.")
            pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
            pc = LidarPointCloud.from_file(pcl_path)
            return pointsensor, pc
        pointsensor_next = nusc.get('sample_data', pointsensor['next'])
        pcl_next_path = osp.join(nusc.dataroot, pointsensor_next['filename'])
        pc_next = LidarPointCloud.from_file(pcl_next_path) 
        return pointsensor_next, pc_next
    elif type == "prev":
        if not pointsensor['prev']:
            # No prev file, return with no change.
            print("Warning: There is no more prev pointcloud file.")
            pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
            pc = LidarPointCloud.from_file(pcl_path)
            return pointsensor, pc
        pointsensor_prev = nusc.get('sample_data', pointsensor['prev'])
        pcl_prev_path = osp.join(nusc.dataroot, pointsensor_prev['filename'])
        pc_prev = LidarPointCloud.from_file(pcl_prev_path)
        return pointsensor_prev, pc_prev   
    
def points2im(points, img):
    h, w = 900, 1600
    im = np.zeros((h, w, 3), dtype=np.uint16)
    points_n = points.shape[1]
    for i in range(points_n):
        x, y = round(points[0, i]), round(points[1, i])
        # if 0<=x<w and 0<=y<h:
        # im[y, x] = round(points[2, i] * 256)
        im[y, x] = img[y, x]
    return im