import torch
from torch import nn
import numpy as np
import tinycudann as tcnn

class NGP(nn.Module):

    def __init__(self, scale, rgb_act="sigmoid"):
    
        super().__init__()

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2) 

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1], where C is levels(?)
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)        # ???
        self.grid_size = 128    # ???
        self.register_buffer('density_bitfield', torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16; N_max = np.log(2048*scale) 
        b = np.exp((N_max/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min}, Nmax={N_max}, b={b:.5f} F={F}, L={L}, T=2^{log2_T}')

        self.xyz_encoder = tcnn.NetworkWithInputEncoding(  
                                            n_input_dims=3, 
                                            n_output_dims=16,
                                            encoding_config={
                                                "otype": "Grid",
                                                "type": "Hash",
                                                "n_levels": L,
                                                "n_features_per_level": F,
                                                "log2_hashmap_size": log2_T,
                                                "base_resolution": N_min,
                                                "per_level_scale": b,
                                                "interpolation": "Linear"
                                            },
                                            network_config={
                                                "otype": "FullyFusedMLP",
                                                "activation": "ReLU",
                                                "output_activation": "None",
                                                "n_neurons": 64,
                                                "n_hidden_layers": 1,
                                            }
            )
        

        self.dir_encoder = tcnn.Encoding(   n_input_dims=3, 
                                            encoding_config={
                                                "otype": "SphericalHarmonics",
                                                "degree": 4,
                                            },
            )
        

        self.rgb_net = tcnn.Network(    n_input_dims=32, n_output_dims=3,
                                        network_config={
                                            "otype": "FullyFusedMLP",
                                            "activation": "ReLU",
                                            "output_activation": self.rgb_act,
                                            "n_neurons": 64,
                                            "n_hidden_layers": 2,
                                        }
        )