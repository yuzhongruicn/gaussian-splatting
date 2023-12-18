import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration
from utils.appearance_utils import AppearanceNet

class AppearanceModel:
    def __init__(self, shs_dim, embed_out_dim, num_views):
        self.appearance_net = AppearanceNet(shs_dim=shs_dim, embed_out_dim=embed_out_dim, num_views=num_views).cuda()
        self.optimizer 	= None

    def step(self, sh_feat, xyz, frame_id, max_batch_size=8192):
    # def step(self, sh_feat, frame_id, max_batch_size=8192):
        B, N = sh_feat.shape[:2]
        if B < max_batch_size:
            d_shs = self.appearance_net(sh_feat, xyz, frame_id)
            # d_shs = self.appearance_net(sh_feat, frame_id)
        else:
            d_shs = torch.empty((B, N), dtype=sh_feat.dtype).cuda()
            head = 0
            while head < B:
                tail = min(head + max_batch_size, B)
                d_shs[head:tail] = self.appearance_net(sh_feat[head:tail], xyz[head:tail], frame_id[head:tail])
                # d_shs[head:tail] = self.appearance_net(sh_feat[head:tail], frame_id[head:tail])
                head += max_batch_size
        # d_shs = self.appearance_net(sh_feat, frame_id)
        return d_shs

    
    def training_setup(self, training_args):
        l = [
            {
                'name': 'Appearance_Model',
                'params': list(self.appearance_net.parameters()),
                'lr': training_args.ap_lr_init,
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.ap_scheduler_args = get_expon_lr_func(lr_init=training_args.ap_lr_init, 
                                                   lr_final=training_args.ap_lr_final, 
                                                   lr_delay_steps=training_args.warm_up,
                                                #    lr_delay_mult=training_args.ap_lr_delay_mult,
                                                    max_steps=training_args.ap_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "appearance_model/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.appearance_net.state_dict(), os.path.join(out_weights_path, 'appearance_model.pth'))

    
    def load_weights(self, model_path, iteration):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "appearance_model"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "appearance_model/iteration_{}/appearance_model.pth".format(loaded_iter))
        self.appearance_net.load_state_dict(torch.load(weights_path))
    
    def updata_lr(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'Appearance_Model':
                param_group['lr'] = self.ap_scheduler_args(iteration)