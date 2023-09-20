import torch
import torchvision.transforms as transforms
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pdb

# params
crop_size = 400

# get image
img = imageio.imread('ca.jpg')
# plt.imshow(img)
# plt.show()

# convert to tensor
img_tensor = torch.Tensor(img).permute(2, 0, 1)
img_cropped = transforms.CenterCrop(crop_size)(img_tensor)
img_flatten = torch.reshape(img_cropped, (crop_size * crop_size, 3))

# create the mesh grid
xy_range = list(map(lambda x: (x / (crop_size - 1) * 2) - 1, range(crop_size)))
xy_range_tensor = torch.Tensor(xy_range)
x_grid, y_grid = torch.meshgrid(xy_range_tensor, xy_range_tensor, indexing='ij')
xy_coord_tensor = torch.stack((x_grid, y_grid), dim= -1)
xy_flatten = torch.reshape(xy_coord_tensor, (crop_size * crop_size, 2))



# pdb.set_trace()


