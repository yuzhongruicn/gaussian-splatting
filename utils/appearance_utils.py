import torch
import torch.nn as nn
import tinycudann as tcnn
import torch.nn.functional as F

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

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

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
    def __init__(self, D=8, W=256, shs_dim=48, embed_out_dim=32):
        super(AppearanceNet, self).__init__()

        self.skips = [D // 2]
        self.t_multires = 10
        self.embed_fn, self.embed_in_dim = get_embedder(self.t_multires, 1)

        self.embed_out_dim = embed_out_dim

        self.embednet = nn.Sequential(
                nn.Linear(self.embed_in_dim, 256), 
                nn.ReLU(inplace=True),
                nn.Linear(256, self.embed_out_dim))
        
        self.linear = nn.ModuleList(
                [nn.Linear(shs_dim + self.embed_out_dim, W)] + 
                [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + shs_dim + self.embed_out_dim, W)
                    for i in range(D - 1)]
            )
        self.output = nn.Linear(W, shs_dim)

    def forward(self, x, fid):
        embedding = self.embednet(self.embed_fn(fid))
        h = torch.cat([x, embedding], dim=-1)
        for i, _ in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, embedding, h], dim=-1)

        d_shs = self.output(h)

        return d_shs
        


    