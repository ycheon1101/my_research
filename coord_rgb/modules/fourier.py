from img_dataset import xy_flatten
import torch.nn as nn
import torch
import math

class GaussianFourier(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale):
        self.xy_flatten = xy_flatten
        self.scale = scale

        # torch[num_input_channels, mapping_size]
        self.B = torch.randn((num_input_channels, mapping_size)) * scale
        # print(f'B shape: {self.B.shape}')
    
    def forward(self, x):
        # calc 2 * pi * B * v
        calc_result = self.B @ x
        calc_result *= 2 * math.pi
        print(f'calc result: {calc_result.shape}')
        return 

    
GaussianFourier(num_input_channels=2, mapping_size=16, scale=10)






