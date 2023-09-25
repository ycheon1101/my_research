from img_dataset import xy_flatten, crop_size
import torch.nn as nn
import torch
import math

class GaussianFourier(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale):
        super().__init__()
        self.xy_flatten = xy_flatten
        self.scale = scale

        # torch[num_input_channels, mapping_size]
        self.B = torch.randn((num_input_channels, mapping_size)) * scale
        # print(f'B shape: {self.B.shape}')
    
    def forward(self, x):
        # calc 2 * pi * B * v
        # make [2, 160000]
        x = x.permute(1, 0)
        calc_result = self.B.T @ x
        calc_result *= 2 * math.pi
        # [16, 160000]
        # print(f'calc result: {calc_result.shape}')
        return torch.cat([torch.cos(calc_result), torch.sin(calc_result)], dim = 0).permute(1, 0)
        
    











