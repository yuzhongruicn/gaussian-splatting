import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import get_expon_lr_func

class AppearanceModel(nn.Module):
    def __init__(self):
        self.optimizer 	= None

    def step(self, sh_feat, time_embedding):
        return
    
    def training_setup(self, training_args):
        l = [
            {
                'name': 'Appearance_Model',
                'params': None,
                'lr': training_args.ap_lr_init,
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.ap_scheduler_args = get_expon_lr_func(lr_init=training_args.ap_lr_init, 
                                                   lr_final=training_args.ap_lr_final, 
                                                   lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.ap_lr_max_steps)

    def save_wrights(self):
        return
    
    def load_wrights(self):
        return
    
    def updata_lr(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'Appearance_Model':
                param_group['lr'] = self.ap_scheduler_args(iteration)