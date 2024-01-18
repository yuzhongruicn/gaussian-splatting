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
import matplotlib.cm as cm


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

def vis_depth(depth):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    percentile = 99
    eps = 1e-10

    lo_auto, hi_auto = weighted_percentile(
        depth, np.ones_like(depth), [50 - percentile / 2, 50 + percentile / 2])
    lo = None or (lo_auto - eps)
    hi = None or (hi_auto + eps)
    curve_fn = lambda x: 1/x + eps

    depth, lo, hi = [curve_fn(x) for x in [depth, lo, hi]]
    depth = np.nan_to_num(
            np.clip((depth - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))
    colorized = cm.get_cmap('turbo')(depth)[:, :, :3]

    return np.uint8(colorized[..., ::-1] * 255)
