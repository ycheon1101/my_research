from img_dataset import img_flatten, xy_flatten, crop_size
# from model import model
from mlp_heatmap import MLP
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from fourier import GaussianFourier
from PIL import Image
import seaborn as sns
# from train import losses_train
import pandas as pd

# params
learning_rate = 5e-3
num_epochs = 300
hidden_feature = 128
hidden_layers = 8
max_pixel = 1.0

# set the target
# target = img_flatten
target = img_flatten

# fourier (mapping size = in_feature / 2)
fourier_result = GaussianFourier(num_input_channels=2, mapping_size = 128, scale=4)(xy_flatten)
# print(f'fourier result shape: {fourier_result.shape}')

net_fourier = MLP(in_feature=256, hidden_feature=hidden_feature, hidden_layers=hidden_layers, out_feature=3)


model_fourier = net_fourier

# calc loss
criterion = nn.MSELoss() 

# Model MLP instanciation

# list for storing MSE losses
# losses = []

# initialize psnr_list
psnr_list = np.zeros((hidden_feature, hidden_layers))

# hit map list
heat_map = []

# hidden_layer_features = [8, 160000, 128]
generated, hidden_layer_features = model_fourier(fourier_result)
# print(f'hidden_layer size: {hidden_layer_features.size()}')
num_layer, num_data, num_neuron = hidden_layer_features.size()

# get data and append to list
for layer in range(num_layer):
    layer_heat_map = []
    for neuron in range(num_neuron):
        heatmap_data = hidden_layer_features[layer, :, neuron]
        layer_heat_map.append(heatmap_data)
    # len(heat_map) : 8, len(heat_map[0]): 128
    heat_map.append(layer_heat_map)

# print(f'heatmap: {len(heat_map[0])}')

# fig, axes = plt.subplots(num_layer, figsize=(10, 8))
# for layer, heatmap_data in enumerate(heat_map):
#     # print(f'heatmap_data: {heatmap_data}')
#     heatmap_data = [data.detach().numpy() for data in heatmap_data]
#     # heatmap_data = torch.tensor(heatmap_data).detach().numpy()
#     sns.heatmap(heatmap_data, ax=axes[layer], cmap="YlGnBu", xticklabels=False, yticklabels=False)
#     axes[layer].set_title(f'Layer {layer+1} Heatmap')

# plt.tight_layout()
# plt.show()

# make pandas table
hidden_layer_df = pd.DataFrame(heat_map)
# print(hidden_layer_df)
sns.heatmap(hidden_layer_df)




# optimizer
optimizer = optim.Adam(model_fourier.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):

    # hidden_layer_features = [8, 160000, 128]
    generated, hidden_layer_features = model_fourier(fourier_result)
    
    # print(f'generate shape: {generated.shape}')
    # print(f'hidden_layer_shape: {hidden_layer_features.shape}')

    # loss = criterion(generated, target)
    loss = criterion(generated, img_flatten)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # losses.append(loss.item())
    # assign psnr to psnr_list
    # for idx_hidden_feature in range(hidden_feature):
    #     for idx_hidden_layers in range(hidden_layers):
    #         calc_psnr = 10 * torch.log10(max_pixel ** 2 / loss)
    #         # print(type(calc_psnr))
    #         psnr_list[idx_hidden_feature, idx_hidden_layers] = calc_psnr


    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# reshape with final generated
generated_reshape = model_fourier(fourier_result)[0]
generated_reshape = torch.reshape(generated_reshape, (crop_size, crop_size, 3))


# for save image
# generated_reshape = generated_reshape * 255.
generated_reshape = generated_reshape.detach().numpy()


# show image
plt.imshow(generated_reshape)
plt.show()

# save image
# image_store = Image.fromarray(generated_reshape.astype(np.uint8))
# image_store.save('scale_4_hidden_128.jpg')

# psnr
# max_pixel = 1.0
# psnr_fourier = [10 * np.log10(max_pixel ** 2 / mse) for mse in losses]
# psnr = [10 * np.log10(max_pixel ** 2 / mse) for mse in losses_train]

# plt.figure(figsize=(10, 5))
# # plt.plot(psnr_list, label="PSNR")
# seaborn.heatmap(psnr_list, xticklabels=hidden_feature, yticklabels=hidden_layers)
# plt.xlabel("hidden_feature")
# plt.ylabel("hidden_layer")
# plt.grid(True)
# plt.title("PSNR")
# plt.show()





