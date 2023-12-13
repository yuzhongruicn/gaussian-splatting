import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.general_utils import get_expon_lr_func
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, ApOptParams, WandbParams, get_combined_args
from models import TimestampEncoder, AppearanceNet
import torch.optim as optim

import wandb

def training(dataset : ModelParams, opt_ap : ApOptParams, pipe : PipelineParams, load_iteration, test_iterations):
    device = torch.device("cuda")
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration)

    train_cameras = scene.getTrainCameras().copy()
    first_frame = train_cameras[0].image_name
    first_timest = float(first_frame.split("_")[1].split(".")[0])
    last_frame = train_cameras[-1].image_name
    last_timest = float(last_frame.split("_")[1].split(".")[0])

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    time_encoder = TimestampEncoder().to(device)
    time_encoder.train()
    ap_net = AppearanceNet().to(device)
    ap_net.train()

    lr_init = opt_ap.lr_init
    optimizer = optim.Adam([
        {"name": "timest_encoder",
         "params": time_encoder.parameters(),
         "lr": lr_init
         },
        {"name": "appearance_net",
         "params": ap_net.parameters(),
         "lr": lr_init   
        }], eps=1e-15)
    lr_scheduler = get_expon_lr_func(lr_init=opt_ap.lr_init, 
                                     lr_final=opt_ap.lr_final,
                                     lr_delay_mult=opt_ap.lr_delay_mult,
                                     max_steps=opt_ap.lr_max_steps)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt_ap.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt_ap.iterations + 1):
        # update lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_scheduler(iteration)

        iter_start.record()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        img_name = viewpoint_cam.image_name
        time_st = (float(img_name.split("_")[1].split(".")[0]) - first_timest) / (last_timest - first_timest)
        time_stamp = torch.tensor([[time_st]], dtype=torch.float32, device="cuda")

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"].detach()
        rendered_feat = render_pkg["rendered_feat"].detach()

        H, W = image.shape[1:3]
        rendered_feat_reshaped = rendered_feat.reshape(rendered_feat.shape[0], -1).transpose(0, 1)
        encoded_time = time_encoder(time_stamp)
        encoded_time = encoded_time.repeat(rendered_feat_reshaped.shape[0], 1)
        # print("encoded_time_: ", encoded_time)
        # print("encoded_time_.shape: ", encoded_time.shape)

        # 
        combined_feat = torch.cat([rendered_feat_reshaped, encoded_time], dim=1)

        out = ap_net(combined_feat)
        # print("out: ", out)
        out = out.reshape(3, H, W)
        image += out
        # print(image)
        # print(image.shape)

        #loss
        gt_image = viewpoint_cam.original_image.cuda()
        if dataset.mask:
            mask = viewpoint_cam.mask.cuda()
            masked_gt_image = torch.where(mask!=True, gt_image, 0).cuda()
            masked_image = torch.where(mask!=True, image, 0).cuda()
            Ll1 = l1_loss(masked_image, masked_gt_image)
            loss = (1.0 - opt_ap.lambda_dssim) * Ll1 + opt_ap.lambda_dssim * (1.0 - ssim(masked_image, masked_gt_image))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt_ap.lambda_dssim) * Ll1 + opt_ap.lambda_dssim * (1.0 - ssim(image, gt_image))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt_ap.iterations:
                progress_bar.close()
        
        # exit()

            # Log
            wandb.log({'train_loss/patches_l1_loss': Ll1.item(),
                       'train_loss/patches_total_loss': loss.item(),
                       'iter_time': iter_start.elapsed_time(iter_end)},
                       step = iteration)
            for param_group in optimizer.param_groups:
                name = param_group["name"]
                lr = param_group['lr']
                wandb.log({f'learning_rates/{name}': lr}, step = iteration)

            if iteration in test_iterations:
                torch.cuda.empty_cache()
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

                            # image = torch.clamp(render(viewpoint, scene.gaussians, pipe, background)["render"], 0.0, 1.0)
                            image = render(viewpoint, gaussians, pipe, background)["render"]
                            rendered_feat = render(viewpoint, gaussians, pipe, background)["rendered_feat"]
                            rendered_feat_reshaped = rendered_feat.reshape(rendered_feat.shape[0], -1).transpose(0, 1)
                            encoded_time = time_encoder(time_stamp)
                            encoded_time = encoded_time.repeat(rendered_feat_reshaped.shape[0], 1)
                            combined_feat = torch.cat([rendered_feat_reshaped, encoded_time], dim=1)
                            out = ap_net(combined_feat)
                            out = out.reshape(3, H, W)
                            image += out
                            image = torch.clamp(image, 0.0, 1.0)
                            
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            if idx < num_log:
                                wandb.log({config['name'] + "_view/{}/render".format(viewpoint.image_name): wandb.Image(image[None], caption="Render")}, step=iteration)
                                image_grid = torch.cat((image, gt_image), dim=-1)
                                wandb.log({config['name'] + "_view/{}/combined".format(viewpoint.image_name): wandb.Image(image_grid[None], caption="Left:Render, Right:GT")}, step=iteration)
                                if iteration == test_iterations[0]:
                                    wandb.log({config['name'] + "_view/{}/GT".format(viewpoint.image_name): wandb.Image(gt_image[None], caption="Ground Truth")}, step=iteration)
                            if dataset.mask: 
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
                        wandb.log({config['name'] + '_loss/viewpoint_l1_loss': l1_test,
                                config['name'] + '_loss/viewpoint_psnr': psnr_test}, 
                                step = iteration)
                torch.cuda.empty_cache()
                    


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = ApOptParams(parser)
    pp = PipelineParams(parser)
    wdb = WandbParams(parser)
    parser.add_argument("--load_iteration", type=int, default=50_000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    #TODO: add a parser for the appear_net params
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing appearance network " + args.model_path)

    # Initialize Wandb
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

    training(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, args.test_iterations)

    # All done
    print("\nTraining complete.")


