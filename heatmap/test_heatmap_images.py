import heatmap_psnr
# import sys
# sys.path.append('/Users/yerin/Desktop/my_research/heatmap/img_data')
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

# def get_image(image):
#     return imageio.imread(image)/255.

# print(get_image('img1').shape)

# def plot_image():

img_path = '/Users/yerin/Desktop/my_research/heatmap/img_data/'
img = imageio.imread(img_path + 'img1.jpeg') / 255.
# [168, 300, 3]
img = torch.tensor(img)
img_permuted = img.permute(2, 0, 1)
# resize
resized_transform = transforms.Resize((200, 200))
resized_img = resized_transform(img_permuted)
print(resized_img.shape)
resized_img = resized_img.permute(1, 2, 0)

plt.imshow(resized_img)
# plt.imshow(img)
plt.axis('off')
plt.show()

# print(img.shape)

