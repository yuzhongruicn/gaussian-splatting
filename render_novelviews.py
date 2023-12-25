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
import numpy as np
from scene import Scene, AppearanceModel
from scene.cameras import Camera
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov, fov2focal



def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, appearance_model=None):
    render_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_feat")
    # makedirs(feat_path, exist_ok=True)

    shs = gaussians.get_features
    xyz = gaussians.get_xyz
    N = shs.shape[0]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if dataset.load2gpu_on_the_fly:
            view.load2device()

        if appearance_model:
            shs = shs.view(N, -1)
            frame_id = view.frame_id
            frame_id = frame_id.unsqueeze(0).expand(N, 1)
            d_shs = appearance_model.step(shs.detach(), xyz.detach(), frame_id)
            d_shs = d_shs.view(-1, (gaussians.max_sh_degree+1)**2, 3)

            render_pkg = render(view, gaussians, pipeline, background, model_appearance=True, d_shs=d_shs)
            rendering = render_pkg["render"]
        else:
            render_pkg = render(view, gaussians, pipeline, background)

        gt = view.original_image[0:3, :, :]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{view.image_name}" + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{view.image_name}" +  ".png"))

        if dataset.load2gpu_on_the_fly:
            view.load2device('cpu')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, free_view : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        num_views = len(scene.getTrainCameras())
        appearance_model = None
        if dataset.add_appearance_embedding:
            appearance_model = AppearanceModel(shs_dim=3*(gaussians.max_sh_degree+1)**2, embed_out_dim=dataset.embedding_dim, num_views=num_views, num_hidden_layers=dataset.ap_num_hidden_layers, num_hidden_neurons=dataset.ap_num_hidden_neurons)
            appearance_model.load_weights(dataset.model_path, iteration=iteration)
    
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if free_view:
            # TODO
            raise NotImplementedError
            # render_poses = scene.getRenderCameras()
            # render_set(dataset, "train", scene.loaded_iter, render_poses, gaussians, pipeline, background)

        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, appearance_model)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, appearance_model)

def get_cam_info(raw_pose, i, embed_id, width, height):

    f = raw_pose["focal_length"]
    FovY = focal2fov(f, height)
    FovX = focal2fov(f, width)
    
    c2w = raw_pose["camera_to_world"]
    c2w[:3, 3] *= 100
    w2c = np.linalg.inv(c2w)

    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    
    camera_entry = {
        'id' : i,
        'img_name' : '{0:05d}'.format(i),
        'width' : width,
        'height' : height,
        'T': T,
        'R': R,
        'FovY' : FovY,
        'FovX' : FovX,
        'frame_id': embed_id
    }
    return camera_entry, raw_pose["origin_pose"]

def get_args(model_path):
    try:
        cfgfilepath = os.path.join(model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    return Namespace(**merged_dict)

def load_cam(camera_entry):

    cam = Camera(colmap_id=camera_entry['id'], R=camera_entry['R'], T=camera_entry['T'], 
                FoVx=camera_entry['R'], FoVy=camera_entry['R'], 
                image=None, gt_alpha_mask=None,
                image_name=camera_entry['img_name'], uid=camera_entry['id'], data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', mask=None, 
                frame_id=camera_entry['frame_id'])
    return cam


def render_img(model_path, camera_entry, embed_num, block, iteration):
    render_path = os.path.join(model_path, "novel_views", block)
    makedirs(render_path, exist_ok=True)

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_args(model_path)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    view = load_cam(dataset, camera_entry)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(os.path.join(model_path,
                                        "point_cloud",
                                        "iteration_" + str(iteration),
                                        "point_cloud.ply"))
        appearance_model = None
        if dataset.add_appearance_embedding:
            appearance_model = AppearanceModel(shs_dim=3*(gaussians.max_sh_degree+1)**2, embed_out_dim=dataset.embedding_dim, num_views=embed_num, num_hidden_layers=dataset.ap_num_hidden_layers, num_hidden_neurons=dataset.ap_num_hidden_neurons)
            appearance_model.load_weights(dataset.model_path, iteration=iteration)
    
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        shs = gaussians.get_features
        xyz = gaussians.get_xyz
        N = shs.shape[0]

        if dataset.load2gpu_on_the_fly:
            view.load2device()

        if appearance_model:
            shs = shs.view(N, -1)
            frame_id = view.frame_id
            frame_id = frame_id.unsqueeze(0).expand(N, 1)
            d_shs = appearance_model.step(shs, xyz, frame_id)
            d_shs = d_shs.view(-1, (gaussians.max_sh_degree+1)**2, 3)

            render_pkg = render(view, gaussians, pipe, background, model_appearance=True, d_shs=d_shs)
        else:
            render_pkg = render(view, gaussians, pipe, background)

        rendered = render_pkg["render"]
        torchvision.utils.save_image(rendered, os.path.join(render_path, f"{view.image_name}" + ".png"))
        return rendered






if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering novel views script parameters")
    parser.add_argument("--all_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    args = parser.parse_args()

    split_path = os.path.join(args.data_path, "json", "idg_split.json")
    with open(split_path) as json_file:
        split_block = json.load(json_file)
    novel_views_path = os.path.join(args.data_path, "json", "novel_view.json")
    with open(novel_views_path) as json_file:
        novel_views = json.load(json_file)
    idx_and_block_path = os.path.join(args.data_path, "json", "index_and_block.json")
    with open(idx_and_block_path) as json_file:
        idx_and_block = json.load(json_file)["index_and_block"]

    blocks = split_block.keys()
    width = novel_views['render_width']
    height = novel_views['render_height']
    novel_view_poses = novel_views['camera_path']
    assert len(novel_view_poses) == len(idx_and_block), "Number of novel views and number of idx do not match"
    for i, raw_pose in enumerate(novel_view_poses):
        block_params = idx_and_block[i]
        print(block_params)
        distances = []
        rendered_images = []
        for embed_id, block in block_params:
            if block is not "block_12":
                continue
            assert block in blocks, "Block not found"
            block_center_pose = split_block[block]['train']['centroid'][1]
            embed_num = len(split_block[block]['train']['elements'])
            camera_entry, origin_pose = get_cam_info(raw_pose, i, embed_id, width, height)
            model_path = os.path.join(args.all_model_path, block)
            rendered = render_img(model_path, camera_entry, embed_num, block, args.iteration).cpu().numpy()
            distance = np.linalg.norm(np.array(camera_entry) - np.array(block_center_pose))
            distances.append(distance)
            rendered_images.append(rendered)

    exit()

        # weights = 1 / np.array(distances)
        # weights /= weights.sum()
        # average_image = np.zeros_like(rendered_images[0])

        # # 对每个图像进行加权平均
        # for img, weight in zip(rendered_images, weights):
        #     average_image += img * weight



