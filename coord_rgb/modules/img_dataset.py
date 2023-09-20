import torch
import torchvision.transforms as transforms
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pdb

# params
crop_size = 400

# get image
img = imageio.imread('ca.jpg')[...,:3]/255.
# plt.imshow(img)
# plt.show()

# convert to tensor
img_tensor = torch.Tensor(img).permute(2, 0, 1)
# print(img_tensor)
img_cropped = transforms.CenterCrop(crop_size)(img_tensor)
img_flatten = torch.reshape(img_cropped, (crop_size * crop_size, 3))


# test
img_crop = img_cropped.permute(1, 2, 0)
# img_crop = img_flatten.view( crop_size, crop_size, 3)


# Now, img_reshaped should have the same shape and values as img_cropped.


# img_crop = torch.reshape(img_cropped, (crop_size, crop_size, 3))
# img_crop =torch.reshape(img_flatten, (crop_size, crop_size, 3))
# img_crop = img_flatten.unsqueeze(0)
# img_crop = img_crop.reshape(img_crop, (crop_size, crop_size, 3))
img_crop.detach().numpy()
# print(img_crop.shape)


plt.imshow(img_crop)
plt.show()

# create the mesh grid
xy_range = list(map(lambda x: (x / (crop_size - 1) * 2) - 1, range(crop_size)))
xy_range_tensor = torch.Tensor(xy_range)
x_grid, y_grid = torch.meshgrid(xy_range_tensor, xy_range_tensor, indexing='ij')
xy_coord_tensor = torch.stack((x_grid, y_grid), dim= -1)
print(f'xy_coord_tensor:{xy_coord_tensor.shape}')
xy_flatten = torch.reshape(xy_coord_tensor, (crop_size * crop_size, 2))



# pdb.set_trace()


