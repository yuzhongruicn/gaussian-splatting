import torch
import torch.nn as nn
import tinycudann as tcnn

class SHEncoding(nn.Module):
    def __init__(self, in_dim : int = 3, levels : int = 4):
        super(SHEncoding, self).__init__()
        self.in_dim = in_dim
        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}")
        self.levels = levels

        encoding_config = self.get_tcnn_encoding_config(levels=self.levels)
        self.tcnn_encoding = tcnn.Encoding(
            n_input_dims=self.in_dim,
            encoding_config=encoding_config
        )

    def get_tcnn_encoding_config(levels):
        encoding_config = {
            "otype": "SphericalHarmonics",
            "levels": levels,
        }
        return encoding_config
    
    def get_out_dim(self):
        return self.levels**2
    
    def foward(self, in_tensor):
        return self.tcnn_encoding(in_tensor)


class Embedding(nn.Module):
    def __init__(self, in_dim : int, out_dim : int):
        super(Embedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.build_nn_modules()

    def build_nn_modules(self):
        self.embedding = nn.Embedding(self.in_dim, self.out_dim)

    def forward(self, in_tensor):
        return self.embedding(in_tensor)

class AppearanceNet(nn.Module):
    def __init__(self):
        super(AppearanceNet, self).__init__()


    