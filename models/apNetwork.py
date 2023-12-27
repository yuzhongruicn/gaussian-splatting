import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

class TimestampEncoder(nn.Module):
    def __init__(self, 
                 timest_input_dim=1,
                 timest_feat_dim=16,
                 hidden_dim_time=32,
                 num_layers_time=3):
        super(TimestampEncoder, self).__init__()
        self.encoder =  tcnn.Network(n_input_dims=timest_input_dim,
                                      n_output_dims=timest_feat_dim,
                                      network_config={
                                          "otype": "CutlassMLP",
                                          "activation": "ReLU",
                                          "output_activation": "None",
                                          "n_neurons": hidden_dim_time,
                                          "n_hidden_layers": num_layers_time
                                      }
        )
    
    def forward(self, x):
        return self.encoder(x)
    

class AppearanceNet(nn.Module):
    def __init__(self,
                 color_feat_dim=48,
                 timest_feat_dim=16,
                 hidden_dim_color=64,
                 num_layers_color=3):
        super(AppearanceNet, self).__init__()
        self.color_net = tcnn.Network(n_input_dims=color_feat_dim + timest_feat_dim,
                                      n_output_dims=3,
                                      network_config={
                                          "otype": "CutlassMLP",
                                          "activation": "ReLU",
                                          "output_activation": "None",
                                          "n_neurons": hidden_dim_color,
                                          "n_hidden_layers": num_layers_color
                                      }

        )

    def forward(self, x):
        output_rgb = self.color_net(x)
        return output_rgb
    
