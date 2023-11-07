'''
transfer kitti dataset to COLMAP format
input: Kitti raw data
'''

import os
import numpy as np 
import argparse
import pykitti
import sqlite3
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def read_args():
    parser = argparse.ArgumentParser(description='transfer kitti dataset to COLMAP format')
    parser.add_argument('-s', '--data_path', type=str, default='/root/paddlejob/workspace/yuzhongrui/datasets')
    parser.add_argument('-o', '--output_path', type=str, default='/root/paddlejob/workspace/yuzhongrui/datasets/2011_09_26_kitti_original')
    parser.add_argument('--pcd', action='store_true', default=False)
    parser.add_argument('--date', type=str, default='2011_09_26')
    parser.add_argument('--drive', type=str, default='0002')

    return parser.parse_args()

if __name__ == "__main__":

    args = read_args()

    basedir =  args.data_path
    date = args.date
    drive = args.drive

    output_dir = args.output_path
    move_img = False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'images'))
        os.makedirs(os.path.join(output_dir, 'sparse'))
        os.makedirs(os.path.join(output_dir, 'sparse', '0'))
        move_img = True

    data = pykitti.raw(basedir, date, drive)
    # print(data.calib)
    # exit()

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

        if move_img:
            cam2_img, cam3_img = data.get_rgb(idx)
            cam2_img.save(os.path.join(output_dir, f'images/02_{idx:010d}.png'))
            # cam3_img.save(os.path.join(output_dir, f'images/03_{idx:010d}.png'))

        if args.pcd:
            points_velo = data.get_velo(idx)
            points_velo[:,-1] = points_velo[:,-1] * 0 + 1

            vel_to_imu = np.linalg.inv(imu_to_vel)
            vel_to_world = np.matmul(imu_to_world, vel_to_imu)
            points_world = np.matmul(vel_to_world, points_velo.T)

            pcd = o3d.geometry.PointCloud()
            pcd.points= o3d.utility.Vector3dVector(points_world.T[:,:3])
            # o3d.io.write_point_cloud(os.path.join(output_dir, f'pcd_{idx}.ply'), pcd)
            if idx == 0:
                points_all = points_world.T[:,:3]
                pcd_all = pcd
            else:
                points_all = np.vstack((points_all, points_world.T[:,:3]))
                pcd_all += pcd


    # f_w = open(os.path.join(output_dir, 'sparse/0/images.txt'), 'w')
    c_w = open(os.path.join(output_dir, 'sparse/0/cameras.txt'), 'w')
    fx = P_rect_20[0, 0]
    fy = P_rect_20[1, 1]
    cx = P_rect_20[0, 2]
    cy = P_rect_20[1, 2]

    for i in range(len(image_names)):
        idx = image_names[i][0]
        name = image_names[i][1]

        transform = cam_to_world[name]
        transform = np.linalg.inv(transform)

        r = R.from_matrix(transform[:3,:3])
        rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
        rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
        out = np.concatenate((rquat, transform[:3, 3]), axis=0)
        # f_w.write(f'{idx + 1} ')
        # f_w.write(' '.join([str(a) for a in out.tolist()] ) )
        # f_w.write(f' {idx + 1} {name}')
        # f_w.write('\n\n')

        c_w.write(f'{idx + 1} PINHOLE 1242 375 {fx} {fy} {cx} {cy}')
        c_w.write('\n')

    # f_w.close()
    c_w.close()

    # o3d.io.write_point_cloud(os.path.join(output_dir,'sparse/0/points3D.ply'), points_all)

    if args.pcd:
        print('# points: ', points_all.shape)
        o3d.io.write_point_cloud(os.path.join(output_dir, 'all.ply'), pcd_all)
        pcd_f_w = open(os.path.join(output_dir,'sparse/0/points3D.txt'), 'w')
        for i, point in enumerate(points_all):
            pcd_f_w.write(f'{i} ')
            pcd_f_w.write(f'{point[0]} {point[1]} {point[2]} ')
            pcd_f_w.write(f'0 0 0 0')
            pcd_f_w.write('\n\n')
        pcd_f_w.close()

        





