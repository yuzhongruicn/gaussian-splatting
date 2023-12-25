import os
import numpy as np 
import argparse
import json
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import shutil
from tqdm import tqdm

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
        # if 'spherical_backward' in key:
        #     continue

        in_K, c2w, H, W = get_camera_in_ex(para_data, key)
        cam_to_world[f'{key}.jpg'] = c2w
        image_names.append(f'{key}.jpg')
        
    f_w = open(os.path.join(output_dir, 'sparse/1/images.txt'), 'w')
    
    print("generating image.txt ...")
    with open(os.path.join(output_dir, 'image_list.txt'), 'r') as l_r:
        lines = l_r.readlines()
        # print(lines)
        for line in tqdm(lines):
            
            image_id, image_name, camera_id = line.strip().split()
            transform = cam_to_world[image_name]
            transform[:3, 3] *= 100
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




