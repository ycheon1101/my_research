import torch
import torchvision.transforms as transforms
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pdb
# from PIL import Image
import numpy as np

# params
crop_size = 400

# get image
# img = imageio.imread('ca.jpg')[...,:3]/255.
# img = imageio.imread('ca.jpg')/255.
img = imageio.imread('ca.jpg') /255.
im = imageio.imread('ca.jpg') /255.
print(f'img shape: {img.shape}')
# print(f'img: {img}')
# img = torch.tensor(img)
# img = torch.reshape(img, (412 * 550, 3))
# plt.imshow(img)
# plt.show()


# im = imageio.imread('ca.jpg')[...,:3]/255.


# im = Image.open('ca.jpg')/255.
# im = Image.open('ca.jpg')
# im = np.array(im)
# print(f'im_np_array: {im}')

# print('same') if np.array_equal(img, im) else print('not')

# convert to tensor
img_tensor = torch.Tensor(img).permute(2, 0, 1)
# print(img_tensor.shape)
img_cropped = transforms.CenterCrop(crop_size)(img_tensor)
print(f'img_cropped shape: {img_cropped.shape}')
img_cropped = img_cropped.permute(1, 2, 0)

# flatten image tensor
img_flatten = torch.reshape(img_cropped,(crop_size * crop_size, 3))

# img_view = img_flatten.reshape(crop_size, crop_size, 3)

# plt.imshow(img_view)
# plt.show()


# test
# img_crop = img_cropped.permute(1, 2, 0)

# img_crop = img_flatten.view( crop_size, crop_size, 3)
# img_crop =torch.reshape(img_flatten, (crop_size, crop_size, 3))

# img_crop = torch.reshape(img_cropped, (crop_size, crop_size, 3))
# img_crop = img_flatten.unsqueeze(0)
# img_crop = img_crop.reshape(img_crop, (crop_size, crop_size, 3))
# img_crop.detach().numpy()
# print(img_crop.shape)


# plt.imshow(img_crop)
# plt.show()

# create the mesh grid
xy_range = list(map(lambda x: (x / (crop_size - 1) * 2) - 1, range(crop_size)))
# print(f'xy_range: {xy_range}')
xy_range_tensor = torch.Tensor(xy_range)
x_grid, y_grid = torch.meshgrid(xy_range_tensor, xy_range_tensor, indexing='ij')
xy_coord_tensor = torch.stack((x_grid, y_grid), dim= -1)
print(f'xy_coord_tensor:{xy_coord_tensor.shape}')
xy_flatten = torch.reshape(xy_coord_tensor, (crop_size * crop_size, 2))



# pdb.set_trace()


