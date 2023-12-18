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
from scene import Scene, AppearanceModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, appearance_model):
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

        # visibility_filter = render(view, gaussians, pipeline, background)["visibility_filter"]
        # shs_visible = shs.view(N, -1)[visibility_filter]
        # xyz_visible = xyz.view(N, -1)[visibility_filter]
        # N_VIS = shs_visible.shape[0]
        # print(view.frame_id)
        # frame_id = view.frame_id.unsqueeze(0).expand(N_VIS, 1)
        # d_shs_vis = appearance_model.step(shs_visible.detach(), xyz_visible.detach(), frame_id)
        # d_shs_vis = d_shs_vis.view(-1, (gaussians.max_sh_degree+1)**2, 3)
        # d_shs = torch.zeros_like(shs)
        # d_shs[visibility_filter] = d_shs_vis
        
        shs = shs.view(N, -1)
        frame_id = view.frame_id
        frame_id = frame_id.unsqueeze(0).expand(N, 1)
        d_shs = appearance_model.step(shs.detach(), xyz.detach(), frame_id)
        d_shs = d_shs.view(-1, (gaussians.max_sh_degree+1)**2, 3)

        render_pkg = render(view, gaussians, pipeline, background, model_appearance=True, d_shs=d_shs)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + f"_{view.image_name}" + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + f"_{view.image_name}" +  ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{view.image_name}" + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{view.image_name}" +  ".png"))

        # rendered_feat = render_pkg["rendered_feat"]
        # torch.save(rendered_feat, os.path.join(feat_path, f"{view.image_name}" + ".pt"))

        if dataset.load2gpu_on_the_fly:
            view.load2device('cpu')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, free_view : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        appearance_model = AppearanceModel(shs_dim=3*(gaussians.max_sh_degree+1)**2, embed_out_dim=32)
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

# def 

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--free_view", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.free_view)