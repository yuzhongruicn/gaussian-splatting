'''
transfer kitti dataset to COLMAP format
input: Kitti raw data
'''

import os
import numpy as np 
import argparse
import json
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import shutil

def read_args():
    parser = argparse.ArgumentParser(description='transfer IDG dataset to COLMAP format')
    parser.add_argument('-s', '--data_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('--block', type=str, default='block_0')

    return parser.parse_args()

def read_intrinsics(f_x,f_y,w,h):
    in_K = np.zeros([3,4])
    in_K[0,0]=f_x
    in_K[1,1]=f_y
    in_K[0,2]=w//2
    in_K[1,2]=h//2
    in_K[2,2]=1
    return in_K

def get_camera_in_ex(param, key):
    H = param[key]['height']
    W = param[key]['width']
    cam2world_temp=np.array(param[key]['transform_matrix'])
    cam2world=np.eye(4)
    cam2world[:3,:]=cam2world_temp
    f_x = param[key]['intrinsics'][0]
    f_y = param[key]['intrinsics'][1]
    in_K = read_intrinsics(f_x,f_y,W,H)
    return in_K, cam2world, H, W

if __name__ == "__main__":

    args = read_args()

    data_path =  args.data_path
    block = args.block
    split_json_file = os.path.join(data_path, 'json', 'idg_split.json')
    para_json_file = os.path.join(data_path, 'json', 'idg_info.json')
    rgb_path = os.path.join(data_path, 'rgb')

    output_dir = args.output_path
    copy_img = False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'images'))
        os.makedirs(os.path.join(output_dir, 'sparse'))
        os.makedirs(os.path.join(output_dir, 'sparse', '0'))
        copy_img = True

    with open(split_json_file, 'r') as f:
        split_data = json.load(f)

    train_split = [x[0] for x in split_data[block]['train']['elements']]
    val_split = [x[0] for x in split_data[block]['val']]
    block_split = train_split + val_split
    block_split.sort()

    with open(para_json_file, 'r') as f:
        para_data = json.load(f)

    cam_to_world = {}
    image_names = []
    intrinsics = {}

    for key in block_split:
        if 'spherical_backward' in key:
            continue
        in_K, c2w, H, W = get_camera_in_ex(para_data, key)
        cam_to_world[f'{key}.jpg'] = c2w
        image_names.append(f'{key}.jpg')
        intrinsics[f'{key}.jpg'] = {'int_K' : in_K, 'height' : H, 'width': W}

        if copy_img:
            img_path = os.path.join(rgb_path, f'{key}.jpg')
            output_img_path = os.path.join(output_dir, 'images', f'{key}.jpg')
            shutil.copyfile(img_path, output_img_path)
            print(f'copy {img_path} to {output_img_path}')

    # f_w = open(os.path.join(output_dir, 'sparse/0/images.txt'), 'w')
    c_w = open(os.path.join(output_dir, 'sparse/0/cameras.txt'), 'w')


    for idx, name in enumerate(image_names):
        # transform = cam_to_world[name]
        # transform = np.linalg.inv(transform)

        # r = R.from_matrix(transform[:3,:3])
        # rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
        # rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
        # out = np.concatenate((rquat, transform[:3, 3]), axis=0)
        # f_w.write(f'{idx + 1} ')
        # f_w.write(' '.join([str(a) for a in out.tolist()] ) )
        # f_w.write(f' {idx + 1} {name}')
        # f_w.write('\n\n')

        # l_w.write(f'{name}')
        # l_w.write('\n')

        in_K = intrinsics[name]['int_K']
        height = intrinsics[name]['height']
        width = intrinsics[name]['width']
        fx = in_K[0,0]
        fy = in_K[1,1]
        cx = int(in_K[0,2])
        cy = int(in_K[1,2])
        c_w.write(f'{idx + 1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}')
        c_w.write('\n')

    c_w.close()




