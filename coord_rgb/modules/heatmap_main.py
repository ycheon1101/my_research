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

# Model MLP instanciation
net_fourier = MLP(in_feature=256, hidden_feature=hidden_feature, hidden_layers=hidden_layers, out_feature=3)
model_fourier = net_fourier


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
        heatmap_data = hidden_layer_features[layer, :, neuron].detach().numpy()
        # calc mean of heatmap data
        heatmap_data = np.mean(heatmap_data)
        layer_heat_map.append(heatmap_data)
    # len(heat_map) : 8, len(heat_map[0]): 128
    heat_map.append(layer_heat_map)

# print(f'heat map type: {type(heat_map)}')

# make pandas table
hidden_layer_df = pd.DataFrame(heat_map)
hidden_layer_df.index = range(1, len(hidden_layer_df) + 1)
hidden_layer_df.columns = range(1, len(hidden_layer_df.columns) + 1)



# plot heatmap

# print(hidden_layer_df)
# sns.heatmap(hidden_layer_df)
plt.figure(figsize=(45, 8))
# sns.heatmap(hidden_layer_df, cmap='coolwarm')
# sns.heatmap(hidden_layer_df, cmap='OrRd')
# sns.heatmap(hidden_layer_df, cmap='GnBu')
# sns.heatmap(hidden_layer_df, cmap='Greys')
# sns.heatmap(hidden_layer_df, cmap='BuPu')

# set x range
xticks_range = list(range(1, hidden_feature + 1, 10))

# put last x val
xticks_range.append(hidden_feature)

heatmap = sns.heatmap(hidden_layer_df, cmap='YlGnBu', xticklabels=True, yticklabels=True)
heatmap.set_xticks(xticks_range)
heatmap.set_xticklabels(xticks_range)
plt.xlabel('neurons')
plt.ylabel('layers')
plt.title('Heatmap')
plt.show()







