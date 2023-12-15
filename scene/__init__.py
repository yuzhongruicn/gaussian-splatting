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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.appearance_model import AppearanceModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if args.data_format == "idg":
            scene_info = sceneLoadTypeCallbacks["IDG"](args.source_path, args.images, args.eval, 
                                                       block=args.block, load_mask=args.mask, 
                                                       spherical_bg=args.spherical_bg, num_bg_points=args.num_bg_points, bg_dist=args.bg_dist)
        elif args.data_format == "colmap" and os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, 
                                                          load_mask=False, 
                                                          spherical_bg=args.spherical_bg, num_bg_points=args.num_bg_points, bg_dist=args.bg_dist)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # if scene_info.bg_path:
            #     with open(scene_info.bg_path, 'rb') as src_file, open(os.path.join(self.model_path, "bg.ply") , 'wb') as dest_file:
            #         dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        
        # if args.data_format == "idg":
        #     first_frame = scene_info.train_cameras[0].image_name
        #     self.first_timest = float(first_frame.split("_")[1].split(".")[0])
        #     last_frame = scene_info.train_cameras[-1].image_name
        #     self.last_timest = float(last_frame.split("_")[1].split(".")[0])

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print('camera_extent: ', self.cameras_extent)

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getRenderCameras(self, scale=1.0, render_path=None):
        train_camera = self.train_cameras[scale][0]
        width = train_camera['width']
        height = train_camera['height']
        fy = train_camera['fy']
        fx = train_camera['fx']

        render_file = render_path if render_path else "render_path.json"
        with open(os.path.join(self.model_path, render_file), 'w') as file:
            render_pose = json.load(file)
        camera_poses = []
        for id, camera_pose in render_pose["camera_path"]:
            c2w = camera_pose["camera_to_world"]
            w2c = np.linalg.inv(c2w)
            pos = w2c[:3, 3]
            rot = w2c[:3, :3]
            serializable_array_2d = [x.tolist() for x in rot]
            camera_entry = {
                'id' : id,
                'img_name' : '{0:05d}'.format(id),
                'width' : width,
                'height' : height,
                'position': pos.tolist(),
                'rotation': serializable_array_2d,
                'fy' : fy,
                'fx' : fx
            }
            camera_poses.append(camera_entry)
        return camera_poses

        