'''
convert open3d pcd to 3D gaussian pcd format
see ./gaussian_splatting/scene/dataset_readers.py
'''

from plyfile import PlyData, PlyElement
import open3d as o3d
import numpy as np
import argparse
import os

def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--input_path', type=str, 
                        default='/root/paddlejob/workspace/yuzhongrui/datasets/2011_09_26_kitti_proj/sparse/0/sparse_all_points75.ply')
    parser.add_argument('-o', '--output_path', type=str, 
                        default='/root/paddlejob/workspace/yuzhongrui/datasets/2011_09_26_kitti_proj/sparse/0')
    parser.add_argument('-ds', '--downsampling', type=str, default='voxel', choices=['voxel','uniform'])
    parser.add_argument('--voxel_size', type = float, default=0.05)
    parser.add_argument('--scale', type=int, default=1)

    return parser.parse_args()

if __name__ == '__main__': 
    args = read_args()
    input_path = args.input_path
    
    pcd_original = o3d.io.read_point_cloud(input_path)
    print('#original points: ', np.asarray(pcd_original.points).shape[0])
    downsample_method = args.downsampling
    
    # uniform downsampling, every k points
    if downsample_method == 'uniform':
        downsample_scale = args.scale
        pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd_original, downsample_scale)
    elif downsample_method == 'voxel':
        voxel_size = args.voxel_size
        # max_bound = pcd_original.get_max_bound()
        # min_bound = pcd_original.get_min_bound()
        pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd_original, voxel_size)
    else:
        raise Exception('Unknown downsampling method: {}'.format(downsample_method))
    
    xyz = np.asarray(pcd.points)
    print('#points: ', xyz.shape[0])
    # exit()
    rgb = np.asarray(pcd.colors) * 255
    rgb = rgb.astype(int)

    output_path = args.output_path

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    if downsample_method == 'uniform':
        ply_data.write(os.path.join(output_path, f'points3D_{downsample_method}_{downsample_scale}.ply'))
    elif downsample_method =='voxel':
        ply_data.write(os.path.join(output_path, f'points3D_{downsample_method}_{voxel_size}.ply'))
    print('sucessfully save point cloud')