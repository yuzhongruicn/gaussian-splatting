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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, AppearanceModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.sh_utils import eval_sh
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, LoggerParams

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False        
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset : ModelParams, opt : OptimizationParams, pipe : PipelineParams, testing_iterations, saving_iterations, logger):
    tb_writer = prepare_output_and_logger(dataset, opt, pipe, logger)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    # num_views = len(scene.getTrainCameras())
    gaussians.training_setup(opt)
    appearance_model = None
    if dataset.add_appearance_embedding:
        appearance_model = AppearanceModel(shs_dim=3*(gaussians.max_sh_degree+1)**2, embed_out_dim=dataset.embedding_dim, num_views=scene.embed_num, num_hidden_layers=dataset.ap_num_hidden_layers, num_hidden_neurons=dataset.ap_num_hidden_neurons)
        appearance_model.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1): 
        torch.cuda.empty_cache()       
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()

        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if dataset.add_appearance_embedding:
            appearance_model.updata_lr(iteration)
            if iteration < opt.warm_up:
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            else:
                model_appearace = True
                shs = gaussians.get_features
                xyz = gaussians.get_xyz
                N = shs.shape[0]
                # only bp to visible gaussians' appearance embedding
                visibility_filter = render(viewpoint_cam, gaussians, pipe, bg)["visibility_filter"]
                shs_visible = shs.view(N, -1)[visibility_filter]
                xyz_visible = xyz.view(N, -1)[visibility_filter]
                N_VIS = shs_visible.shape[0]
                frame_id = viewpoint_cam.frame_id.unsqueeze(0).expand(N_VIS, 1)
                d_shs_vis = appearance_model.step(shs_visible, xyz_visible.detach(), frame_id)
                d_shs_vis = d_shs_vis.view(-1, (gaussians.max_sh_degree+1)**2, 3)
                d_shs = torch.zeros_like(shs)
                d_shs[visibility_filter] = d_shs_vis

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, model_appearace, d_shs)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if dataset.mask:
            mask = viewpoint_cam.mask.cuda()
            masked_gt_image = torch.where(mask!=True, gt_image, 0).cuda()
            masked_image = torch.where(mask!=True, image, 0).cuda()
            Ll1 = l1_loss(masked_image, masked_gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_image, masked_gt_image))
            loss.backward()
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if logger == "wandb" and WANDB_FOUND:
                training_report_wandb(dataset, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), gaussians, appearance_model)
            else:
                training_report(tb_writer, dataset, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), gaussians, appearance_model)
      
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if args.add_appearance_embedding:
                    appearance_model.save_weights(scene.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if args.add_appearance_embedding:
                    appearance_model.optimizer.step()
                    appearance_model.optimizer.zero_grad()

def prepare_output_and_logger(args, opt, pipe, logger):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Save other params for reproducibility
    with open(os.path.join(args.model_path, "opt_args"), 'w') as opt_log_f:
        opt_log_f.write(str(Namespace(**vars(opt))))
    with open(os.path.join(args.model_path, "pipe_args"), 'w') as pipe_log_f:
        pipe_log_f.write(str(Namespace(**vars(pipe))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and logger == "tensorboard":
        tb_writer = SummaryWriter(args.model_path)
    elif WANDB_FOUND and logger == "wandb":
        print("Logging to Wandb")
    else:
        print("no logging progress")
    return tb_writer

def training_report(tb_writer, args, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gaussians : GaussianModel, appearance_model=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        for param_group in gaussians.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = param_group['lr']
                tb_writer.add_scalar('learning_rates/xyz', lr, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        shs = scene.gaussians.get_features
        xyz = scene.gaussians.get_xyz
        N = shs.shape[0]
        shs = shs.view(N, -1)

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if args.load2gpu_on_the_fly:
                        viewpoint.load2device()

                    if args.add_appearance_embedding:
                        frame_id = viewpoint.frame_id
                        frame_id = frame_id.unsqueeze(0).expand(N, 1)
                        # d_shs = appearance_model.step(shs.detach(), frame_id)
                        d_shs = appearance_model.step(shs.detach(), xyz.detach(), frame_id)
                        d_shs = d_shs.view(-1, (gaussians.max_sh_degree+1)**2, 3)
                        image =renderFunc(viewpoint, scene.gaussians, *renderArgs, model_appearance=True, d_shs=d_shs)["render"]
                        
                    else:
                        image =renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]

                    gt_image = viewpoint.original_image.to("cuda")
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    if args.mask: 
                        mask = viewpoint.mask.cuda()
                        masked_gt_image = torch.where(mask!=True, gt_image, 0).cuda()
                        masked_image = torch.where(mask!=True, image, 0).cuda()
                        l1_test += l1_loss(masked_image, masked_gt_image).mean().double()
                        psnr_test += psnr(masked_image, masked_gt_image).mean().double()
                    else:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if args.load2gpu_on_the_fly:
                    viewpoint.load2device('cpu')
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def training_report_wandb(args,iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gaussians : GaussianModel, appearance_model=None):
    # wandb.watch(appearance_model.appearance_net, log="all") 
    wandb.log({'train_loss/patches_l1_loss': Ll1.item(),
               'train_loss/patches_total_loss': loss.item(),
               'iter_time': elapsed},
               step = iteration)
    for param_group in gaussians.optimizer.param_groups:
        if param_group["name"] == "xyz":
            lr = param_group['lr']
            wandb.log({'learning_rates/xyz': lr}, step = iteration)
    if args.add_appearance_embedding:
        for param_group in appearance_model.optimizer.param_groups:
            if param_group["name"] == "Appearance_Model":
                lr = param_group['lr']
                wandb.log({'learning_rates/appearance_model': lr}, step = iteration)
    # Report #points at the beginnig of the training
    if iteration == 1:
        wandb.log({"scene/total_points": scene.gaussians.get_xyz.shape[0]}, step = iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        shs = scene.gaussians.get_features
        xyz = scene.gaussians.get_xyz
        N = shs.shape[0]
        shs = shs.view(N, -1)

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                if config['name'] == "test":
                    num_log = len(config['cameras'])        # log all test images
                else:
                    num_log = 5
                for idx, viewpoint in enumerate(config['cameras']):

                    if args.load2gpu_on_the_fly:
                        viewpoint.load2device()
                    if args.add_appearance_embedding:
                        frame_id = viewpoint.frame_id
                        frame_id = frame_id.unsqueeze(0).expand(N, 1)
                        # d_shs = appearance_model.step(shs.detach(), frame_id)
                        d_shs = appearance_model.step(shs.detach(), xyz.detach(), frame_id)
                        d_shs = d_shs.view(-1, (gaussians.max_sh_degree+1)**2, 3)
                        image =renderFunc(viewpoint, scene.gaussians, *renderArgs, model_appearance=True, d_shs=d_shs)["render"]
                        
                    else:
                        image =renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]

                    gt_image = viewpoint.original_image.to("cuda")


                    if idx < num_log:
                        wandb.log({config['name'] + "_view/{}/render".format(viewpoint.image_name): wandb.Image(image[None], caption="Render")}, step=iteration)
                        image_grid = torch.cat((image, gt_image), dim=-1)
                        wandb.log({config['name'] + "_view/{}/combined".format(viewpoint.image_name): wandb.Image(image_grid[None], caption="Left:Render, Right:GT")}, step=iteration)
                        if iteration == testing_iterations[0]:
                            wandb.log({config['name'] + "_view/{}/GT".format(viewpoint.image_name): wandb.Image(gt_image[None], caption="Ground Truth")}, step=iteration)
                    if args.mask: 
                        mask = viewpoint.mask.cuda()
                        masked_gt_image = torch.where(mask!=True, gt_image, 0).cuda()
                        masked_image = torch.where(mask!=True, image, 0).cuda()
                        l1_test += l1_loss(masked_image, masked_gt_image).mean().double()
                        psnr_test += psnr(masked_image, masked_gt_image).mean().double()
                    else:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if args.load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                wandb.log({config['name'] + '_loss/viewpoint_l1_loss': l1_test,
                           config['name'] + '_loss/viewpoint_psnr': psnr_test}, 
                           step = iteration)

        wandb.log({"scene/opacity_histogram": wandb.Histogram(scene.gaussians.get_opacity.cpu()),
                   "scene/total_points": scene.gaussians.get_xyz.shape[0]}, 
                   step = iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    wdb = LoggerParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--logger", type=str, default = None, choices=[None, "wandb", "tensorboard"])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Initialize Wandb
    if args.logger == "wandb":
        assert WANDB_FOUND, "Wandb not found. Please install it with pip."
        wdb = wdb.extract(args)
        if wdb.wandb_disabled:
            mode = "disabled"
        else:
            mode = "offline"        # set offline on pdc and sync in docker
        if wdb.run_name == None:
            wandb.init(project=wdb.project_name,
                    mode=mode)
        else:
            wandb.init(project=wdb.project_name,
                    name=wdb.run_name,
                    mode=mode)
        wandb.config.update(args)
    elif args.logger == "tensorboard":
        assert TENSORBOARD_FOUND, "Tensorboard not found. Please install it with pip."
    else:
        WANDB_FOUND = False 
        TENSORBOARD_FOUND = False

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.logger)

    # All done
    print("\nTraining complete.")
