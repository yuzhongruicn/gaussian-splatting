#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY, fx, fy, cx, cy, W, H):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    
    
    # top = tanHalfFovY * znear
    # bottom = -top
    # right = tanHalfFovX * znear
    # left = -right
    # print('fx:', fx, 'fy:', fy, 'cx:', cx, 'cy:', cy, 'W:', W, 'H:', H)

    P = torch.zeros(4, 4)

    z_sign = 1.0

    # P[0, 0] = 2.0 * znear / (right - left)
    # P[1, 1] = 2.0 * znear / (top - bottom)
    # P[0, 2] = (right + left) / (right - left)
    # P[1, 2] = (top + bottom) / (top - bottom)
    # P[3, 2] = z_sign
    # # P[2, 2] = z_sign * zfar / (zfar - znear)
    # # P[2, 3] = -(zfar * znear) / (zfar - znear)
    # P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    # P[2, 3] = -2.0 * (zfar * znear) / (zfar - znear)
    
    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = 2 * (cx / W) - 1.0
    P[1, 2] = 2 * (cy / H) - 1.0
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[3, 2] = z_sign
    P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# 弧度
def rotationMatrixToEulerAngles(R):
    if type(R) != 'numpy':
        R = np.array(R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([y, x, z])

# 角度转弧度
def angle2radian(theta):
    return [i * math.pi / 180.0 for i in theta]

# 弧度转角度
def radian2angle(theta):
    return [i * 180.0/ math.pi for i in theta]

# 欧拉角（默认为弧度）转旋转矩阵
def eulerAngles2rotationMat(theta, format='radian'):
    if format =='angle':
        theta = angle2radian(theta)
    theta[0], theta[1] = theta[1], theta[0]    
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R