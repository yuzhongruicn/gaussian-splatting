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

from scene.cameras import Camera, PseudoCamera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import math
from utils.graphics_utils import rotationMatrixToEulerAngles, radian2angle, angle2radian, eulerAngles2rotationMat
from utils.graphics_utils import focal2fov, fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]

    # 忽略这个gt_alpha_mask
    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    mask = None
    if args.mask:
        assert cam_info.mask is not None, f"No mask for camera {cam_info.image_name}"
        mask = PILtoTorch(cam_info.mask, resolution)
    
    cx = cam_info.cx / (orig_w / resolution[0])
    cy = cam_info.cy / (orig_h / resolution[1])
    
    fx = cam_info.fx / (orig_w / resolution[0])
    fy = cam_info.fy / (orig_h / resolution[1])

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, cx=cx, cy=cy, fx=fx, fy=fy,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', mask=mask, frame_id=cam_info.frame_id)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        # 'fy' : fov2focal(camera.FovY, camera.height),
        # 'fx' : fov2focal(camera.FovX, camera.width),
        'fx' : camera.fx,
        'fy' : camera.fy,
        'cx' : camera.cx,
        'cy' : camera.cy
    }
    return camera_entry

def generate_random_cams_kitti360(cameras, num_frames=1000):
    train_cam = cameras[0]
    fx, fy = train_cam.fx, train_cam.fy
    
    #TODO not hard code
    height = train_cam.image_height
    width = 600
    cx = width / 2
    cy = height / 2
    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)

    np.random.seed(42)
    delta_angle = np.random.randint(-60, 60, num_frames)
    
    # poses = []
    pseudo_cams = []
    #TODO: generate random poses
    for idx, camera in enumerate(cameras[:-1]):
        img_name = camera.image_name
        w2c = np.eye(4)
        w2c[:3, :3] = camera.R.transpose()
        w2c[:3, 3] = camera.T
        c2w = np.linalg.inv(w2c)
        
        # w2c_next = np.eye(4)
        # camera_next = cameras[idx+1]
        # w2c_next[:3, :3] = camera_next.R.T
        # w2c_next[:3, 3] = camera_next.T
        # c2w_next = np.linalg.inv(w2c_next)
        
        c2w_new = np.eye(4)
        # c2w_new[:3, 3] = 0.5 * (c2w_next[:3, 3] + c2w[:3, 3])
        c2w_new[:3, 3] = c2w[:3, 3]
        euler = rotationMatrixToEulerAngles(c2w[:3, :3])
        angle = radian2angle(euler)
        
        #（-60 - 60）
        angle[2] += delta_angle[idx]
        rot = eulerAngles2rotationMat(angle2radian(angle))
        
        c2w_new[:3, :3] = rot
        w2c_new = np.linalg.inv(c2w_new)
        # poses.append(w2c_new)
        pseudo_cams.append(PseudoCamera(R=w2c_new[:3, :3].T, T=w2c_new[:3, 3], FoVx=FoVx, FoVy=FoVy,
                                        cx=cx, cy=cy, fx=fx, fy=fy,
                                        width=width, height=height, image_name=f"{img_name}_{delta_angle[idx]}"))
        
    return pseudo_cams