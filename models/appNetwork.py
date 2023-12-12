import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

class TimestampEncoder(nn.Module):
    def __init__(self, timest_feat_dim=16):
        super(TimestampEncoder, self).__init__()
        self.encoder = nn.Linear(1, timest_feat_dim)
    
    def forward(self, x):
        return self.encoder(x)
    

class AppearanceNet(nn.Module):
    def __init__(self,
                 color_feat_dim,
                 timest_feat_dim=16,
                 hidden_dim_color=64,
                 num_layers_color=3):
        super(AppearanceNet, self).__init__()
        self.timestamp_encoder = TimestampEncoder(timest_feat_dim)
        self.color_net = tcnn.Network(n_input_dims=color_feat_dim + timest_feat_dim,
                                      n_output_dims=3,
                                      network_config={
                                          "otype": "FullyFusedMLP",
                                          "activation": "ReLU",
                                          "output_activation": "None",
                                          "n_neurons": hidden_dim_color,
                                          "n_hidden_layers": num_layers_color
                                      }

        )

    def forward(self, timestamps, sh_feature):
        encoded_timestamps = self.timestamp_encoder(timestamps)
        x = torch.cat((encoded_timestamps, sh_feature), dim=1)
        output_rgb = self.color_net(x)
        return output_rgb
    
