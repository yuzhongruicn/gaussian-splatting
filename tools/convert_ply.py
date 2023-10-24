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
    return parser.parse_args()

if __name__ == '__main__': 
    args = read_args()
    input_path = args.input_path
    
    pcd = o3d.io.read_point_cloud(input_path)
    xyz = np.asarray(pcd.points)
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
    ply_data.write(os.path.join(output_path, 'point3Ds.ply'))