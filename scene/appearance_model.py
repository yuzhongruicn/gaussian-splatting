import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration
from utils.appearance_utils import AppearanceNet

class AppearanceModel:
    def __init__(self, shs_dim, embed_out_dim):
        self.appearance_net = AppearanceNet(shs_dim=shs_dim, embed_out_dim=embed_out_dim).cuda()
        self.optimizer 	= None

    def step(self, sh_feat, frame_id):
        return self.appearance_net(sh_feat, frame_id)
    
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
                                                   lr_delay_mult=training_args.ap_lr_delay_mult,
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