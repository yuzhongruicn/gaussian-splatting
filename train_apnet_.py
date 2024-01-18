import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
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
HEIGHT = 930
WIDTH = 1440

# class gsRenderedDataset(Dataset):
#     def __init__(self, image_path, feat_path, gt_path, first_timest, last_timest, transform=None):
#         self.image_path = image_path
#         self.feat_path = feat_path
#         self.gt_path = gt_path
#         self.image_list = sorted(os.listdir(image_path))
#         self.first_timest = first_timest
#         self.last_timest = last_timest
        
#     def __len__(self):
#         return len(self.image_list)
    
#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         time_st = (float(time_stamp.split("_")[1].split(".")[0]) - self.first_timest) / (self.last_timest - self.first_timest)
#         time_stamp = torch.tensor([[time_st]], dtype=torch.float32, device="cuda")

#         image = Image.open(os.path.join(self.image_path, image_name+'.png')).convert("RGB")
#         image = self.transform(image).to(device='cuda')
#         feat = torch.load(os.path.join(self.feat_path, image_name+'.pt')).to(device='cuda')
#         gt = Image.open(os.path.join(self.gt_path, image_name+'.png')).convert("RGB")
#         gt = self.transform(gt).to(device='cuda')
#         return image, feat, time_stamp, gt

def training(dataset : ModelParams, opt_ap : ApOptParams, pipe : PipelineParams, load_iteration, test_iterations, batch_size=4096, batch_img_num=20):
    device = torch.device("cuda")
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration)

    # only for IDG data format
    first_timest = scene.first_timest
    last_timest = scene.last_timest
    print("first_timest: ", first_timest)
    print("last_timest: ", last_timest)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    time_encoder = TimestampEncoder().to(device)
    time_encoder.train()
    ap_net = AppearanceNet().to(device)
    ap_net.train()

    wandb.watch(time_encoder, log="all")
    wandb.watch(ap_net, log="all")

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

        # create a batch of images
        batch_feats = torch.empty((batch_size*batch_img_num, 64), device=device)
        gt_images = torch.empty((batch_size*batch_img_num, 3), device=device)
        for batch_idx in range(0, batch_img_num):
        # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            img_name = viewpoint_cam.image_name
            time_st = (float(img_name.split("_")[1].split(".")[0]) - first_timest) / (last_timest - first_timest)
            time_stamp = torch.tensor([[time_st]], dtype=torch.float32, device="cuda")

            render_pkg = render(viewpoint_cam, scene.gaussians, pipe, background)
            image = render_pkg["render"].detach()
            rendered_feat = render_pkg["rendered_feat"].detach()
            # image = Image.open(os.path.join(dataset.model_path, 'train', f'ours_{load_iteration}', 'renders', img_name+'.png')).convert("RGB")
            # image = transform(image).to(device)
            # rendered_feat = torch.load(os.path.join(dataset.model_path, 'train', f'ours_{load_iteration}', 'rendered_feat', img_name+'.pt')).to(device)

            H, W = image.shape[1:3]
            rendered_feat_reshaped = rendered_feat.reshape(rendered_feat.shape[0], -1).transpose(0, 1)
            encoded_time = time_encoder(time_stamp)
            encoded_time = encoded_time.repeat(rendered_feat_reshaped.shape[0], 1)
            combined_feat = torch.cat([rendered_feat_reshaped, encoded_time], dim=1)

            indices = torch.randperm(combined_feat.shape[0])[:batch_size]
            batch_feats[batch_idx*batch_size:(batch_idx+1)*batch_size] = combined_feat[indices]

            gt_image = viewpoint_cam.original_image
            gt_image = gt_image.reshape(H*W, 3)
            gt_images[batch_idx*batch_size:(batch_idx+1)*batch_size] = gt_image[indices].cuda()

        pred_images = ap_net(batch_feats)
        Ll1 = l1_loss(pred_images, gt_images)
        loss = Ll1  
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

            # Log
            training_report_wandb(iteration, Ll1, loss, iter_start.elapsed_time(iter_end), test_iterations, scene, (pipe, background), optimizer, first_timest, last_timest, time_encoder, ap_net, dataset.mask)


def training_report_wandb(iteration, Ll1, loss, elapsed, testing_iterations, scene : Scene, renderArgs, optimizer, first_timest, last_timest, time_encoder, ap_net, load_mask=False):
    wandb.log({'train_loss/patches_l1_loss': Ll1.item(),
                'train_loss/patches_total_loss': loss.item(),
                'iter_time': elapsed},
                step = iteration)
    for param_group in optimizer.param_groups:
        name = param_group["name"]
        lr = param_group['lr']
        wandb.log({f'learning_rates/{name}': lr}, step = iteration)

    if iteration in testing_iterations:
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

                    img_name = viewpoint.image_name
                    time_st = (float(img_name.split("_")[1].split(".")[0]) - first_timest) / (last_timest - first_timest)
                    time_stamp = torch.tensor([[time_st]], dtype=torch.float32, device="cuda")

                    image = render(viewpoint, scene.gaussians, *renderArgs)["render"]
                    # image = render(viewpoint, gaussians, pipe, background)["render"]
                    rendered_feat = render(viewpoint, scene.gaussians, *renderArgs)["rendered_feat"]
                
                    rendered_feat_reshaped = rendered_feat.reshape(rendered_feat.shape[0], -1).transpose(0, 1)
                    encoded_time = time_encoder(time_stamp)
                    encoded_time = encoded_time.repeat(rendered_feat_reshaped.shape[0], 1)
                    combined_feat = torch.cat([rendered_feat_reshaped, encoded_time], dim=1)
                    out = ap_net(combined_feat)
                    out = out.reshape(3, HEIGHT, WIDTH)
                    # print(image.shape, out.shape)
                    # out = torch.clamp(out, 0.0, 1.0)
                    # image_combine = opt_ap.lambda_img * image + (1 - opt_ap.lambda_img * out)
                    image_final = image + out
                    gt_image = viewpoint.original_image.to("cuda")
                    if idx < num_log:
                        wandb.log({config['name'] + "_view/{}/render".format(viewpoint.image_name): wandb.Image(image[None], caption="GS-Render")}, step=iteration)
                        wandb.log({config['name'] + "_view/{}/residual".format(viewpoint.image_name): wandb.Image(out[None], caption="Residual")}, step=iteration)
                        wandb.log({config['name'] + "_view/{}/final".format(viewpoint.image_name): wandb.Image(image_final[None], caption="Final")}, step=iteration)                                
                        image_grid = torch.cat((image, image_final, gt_image), dim=-1)
                        wandb.log({config['name'] + "_view/{}/combined".format(viewpoint.image_name): wandb.Image(image_grid[None], caption="Left:GS-Render, Mid:Final, Right:GT")}, step=iteration)
                        
                        # image_grid = torch.cat((image_final, gt_image), dim=-1)
                        # wandb.log({config['name'] + "_view/{}/combined".format(viewpoint.image_name): wandb.Image(image_grid[None], caption="Left:Render, Right:GT")}, step=iteration)
                        if iteration == testing_iterations[0]:
                            gt_residual = gt_image - image
                            wandb.log({config['name'] + "_view/{}/GT-Res".format(viewpoint.image_name): wandb.Image(gt_residual[None], caption="GT-Res")}, step=iteration)
                            wandb.log({config['name'] + "_view/{}/GT".format(viewpoint.image_name): wandb.Image(gt_image[None], caption="Ground Truth")}, step=iteration)
                    if load_mask: 
                        mask = viewpoint.mask.cuda()
                        masked_gt_image = torch.where(mask!=True, gt_image, 0).cuda()
                        masked_image = torch.where(mask!=True, image_final, 0).cuda()
                        l1_test += l1_loss(masked_image, masked_gt_image).mean().double()
                        psnr_test += psnr(masked_image, masked_gt_image).mean().double()
                    else:
                        l1_test += l1_loss(image_final, gt_image).mean().double()
                        psnr_test += psnr(image_final, gt_image).mean().double()
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


