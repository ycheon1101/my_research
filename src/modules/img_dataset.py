import torch
import torchvision.transforms as transforms
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# initialize
crop_size = 400


def img_data(img_path, img_name):

    # get image
    img = imageio.imread(img_path + img_name) / 255.
    # convert to tensor [c, h, w]
    img_tensor = torch.Tensor(img).permute(2, 0, 1)
    # crop or resize image
    # img_cropped = transforms.CenterCrop(crop_size)(img_tensor)
    img_cropped = transforms.Resize((crop_size, crop_size),antialias=False)(img_tensor)
    # permute to [h, w, c]
    img_cropped = img_cropped.permute(1, 2, 0)
    # flatten image tensor
    img_flatten = torch.reshape(img_cropped,(crop_size * crop_size, 3))
    # create the mesh grid
    xy_range = list(map(lambda x: (x / (crop_size - 1) * 2) - 1, range(crop_size)))
    xy_range_tensor = torch.Tensor(xy_range)
    x_grid, y_grid = torch.meshgrid(xy_range_tensor, xy_range_tensor, indexing='ij')
    xy_coord_tensor = torch.stack((x_grid, y_grid), dim= -1)
    xy_flatten = torch.reshape(xy_coord_tensor, (crop_size * crop_size, 2))

    return img, crop_size, img_flatten, xy_flatten, img_cropped

# img_data('/Users/yerin/Desktop/my_research/src/images/heatmap_test/', 'img1.jpeg')

