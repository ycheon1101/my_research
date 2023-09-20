import torch
import imageio.v2 as imageio
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
import itertools
import math
import torch.optim as optim
import torch.nn.functional as F


####### Create a small MLP ##########
# class MLP(torch.nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: int,
#         hidden_layers: int,
#         out_features: int
#     ):
#         super().__init__()
#         out_features = out_features if out_features is not None else hidden_features
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(in_features, hidden_features),
#             torch.nn.ReLU(),
#             *[torch.nn.Sequential(
#                 torch.nn.Linear(hidden_features, hidden_features),
#                 torch.nn.ReLU()
#             ) for _ in range(hidden_layers)],
#             torch.nn.Linear(hidden_features, out_features)
#         )
#     def forward(self, x: torch.Tensor):
#         return self.net(x)



################### MAIN ###############

def main():

    #0. params 
    crop_size = 256
    nb_epochs = 100

    #1. read the image
    im = imageio.imread('ca.jpg')

    #2. convert to tensor
    im_tensor = torch.Tensor(im).permute(2, 0, 1)
    my_crop = transforms.CenterCrop(crop_size)
    im_crop = my_crop(im_tensor)

    # 3. flatten tensor to [crop_size**2, 3]
    im_flat = torch.reshape(im_crop,(crop_size*crop_size, 3))

    #4. Create the mesh grid
    xy_range = list(map(lambda x: x / crop_size, range(0, crop_size)))
    xy_range_tensor = torch.tensor(xy_range, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(xy_range_tensor, xy_range_tensor)
    xy_coord_tensor = torch.stack((x_grid, y_grid), dim=-1)
    xy_flat = torch.reshape(xy_coord_tensor,(crop_size*crop_size, 2))

    # 5. Declare network
    net = MLP(in_features=2, hidden_features=256, hidden_layers=2, out_features=3)

    # 6. Try one forward
    #recons_im = torch.reshape(net(xy_flat), (crop_size, crop_size,3))
    # loss = net(xy_flat) - im_flat


    ########## TRAIN!
    # for epoch in range(nb_epochs):
    #     XXXX = 1 

main()

