import os
import numpy as np 
import argparse
import json
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import shutil
import pykitti

def read_args():
    parser = argparse.ArgumentParser(description='transfer IDG dataset to COLMAP format')
    parser.add_argument('-s', '--data_path', type=str)
    parser.add_argument('--image_list_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('--date', type=str, default='2011_09_26')
    parser.add_argument('--drive', type=str, default='0002')


    return parser.parse_args()


if __name__ == "__main__":

    args = read_args()

    data_path =  args.data_path
    image_list_path = args.image_list_path
    date = args.date
    drive = args.drive

    output_dir = args.output_path

    data = pykitti.raw(data_path, date, drive)

    # load intrinsics
    P_rect_20 = data.calib.P_rect_20
    R_rect_20 = data.calib.R_rect_20
    
    P_rect_30 = data.calib.P_rect_30
    R_rect_30 = data.calib.R_rect_30

    # load extrinsics
    imu_to_cam2 = data.calib.T_cam2_imu
    velo_to_cam2 = data.calib.T_cam2_velo

    imu_to_cam3 = data.calib.T_cam3_imu
    velo_to_cam3 = data.calib.T_cam3_velo

    imu_to_vel = data.calib.T_velo_imu
    
    cam_to_world = {}
    image_names = []

    for idx, _ in enumerate(data.oxts):
        # continue
        imu_to_world = data.oxts[idx].T_w_imu
        # print(imu_to_world)

        cam2_to_imu = np.linalg.inv(imu_to_cam2)
        cam2_to_world = np.matmul(np.matmul(imu_to_world, cam2_to_imu), np.linalg.inv(R_rect_20))

        cam3_to_imu = np.linalg.inv(imu_to_cam3)
        cam3_to_world = np.matmul(np.matmul(imu_to_world, cam3_to_imu), np.linalg.inv(R_rect_30))

        cam_to_world[f'02_{idx:010d}.png'] = cam2_to_world
        cam_to_world[f'03_{idx:010d}.png'] = cam3_to_world
        image_names.append([idx ,f'02_{idx:010d}.png'])
        # image_names.append('03_{idx:010d}.png')
        
    f_w = open(os.path.join(output_dir, 'sparse/0/images.txt'), 'w')
    print(image_list_path)
    
    with open(os.path.join(image_list_path, 'image_list.txt'), 'r') as l_r:
        lines = l_r.readlines()
        print(lines)
        for line in lines:
            image_id, image_name, camera_id = line.strip().split()
            transform = cam_to_world[image_name]
            transform = np.linalg.inv(transform)
            r = R.from_matrix(transform[:3,:3])
            rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
            rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
            out = np.concatenate((rquat, transform[:3, 3]), axis=0)
            f_w.write(f'{image_id} ')
            f_w.write(' '.join([str(a) for a in out.tolist()] ) )
            f_w.write(f' {camera_id} {image_name}')
            f_w.write('\n\n')


    f_w.close()
    l_r.close()




