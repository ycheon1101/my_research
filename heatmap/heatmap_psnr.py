import sys
sys.path.append('/Users/yerin/Desktop/my_research/coord_rgb/modules')
from img_dataset import img_flatten, xy_flatten, crop_size
# import img_dataset
# from model import model
from fourier_mlp import MLP
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from fourier import GaussianFourier
from PIL import Image
import seaborn as sns
import pandas as pd


def main():

    # params
    learning_rate = 5e-3
    num_epochs = 300
    hidden_feature = 128
    hidden_layers = 8
    max_pixel = 1.0


    # create layer and neuron list
    neuron_list = [16, 32, 64, 128, 256, 512]
    hidden_layer_list = [1, 2, 4, 6, 8, 10]
    # neuron_list = [16, 32]
    # hidden_layer_list = [1, 2]



    # set the target
    # target shape = [crop_size * crop_size, 3]
    target = img_flatten
    # print(target.shape)

    # fourier (mapping size = in_feature / 2)
    fourier_result = GaussianFourier(num_input_channels=2, mapping_size = 128, scale=4)(xy_flatten)

    # calc loss
    criterion = nn.MSELoss() 

    # initialize psnr_list 
    psnr_list = np.zeros((len(neuron_list), len(hidden_layer_list)))

    # train for num_neuron * num_hidden_layer
    for layer in range(len(hidden_layer_list)):
        for neuron in range(len(neuron_list)):

            net = MLP(in_feature=256, hidden_feature=neuron_list[neuron], hidden_layers=hidden_layer_list[layer], out_feature=3)

            model = net

            optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

            for epoch in range(num_epochs):

                generated = model(fourier_result)
                loss = criterion(generated, img_flatten)

                # optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # calc psnr
            max_pixel = 1.0
            calc_psnr = 10 * torch.log10(max_pixel ** 2 / loss)
            psnr_list[layer, neuron] = calc_psnr

            print(f"layer: {hidden_layer_list[layer]}, neuron: {neuron_list[neuron]}, Loss: {loss.item()}, calc_psnr: {calc_psnr}")

            # Getting the output from the network and reshaping it
            generated_img = model(fourier_result)
            generated_reshape = torch.reshape(generated_img, (crop_size, crop_size, 3))

            generated_reshape = generated_reshape * 255.0
            generated_reshape = generated_reshape.detach().numpy()

            save_img = Image.fromarray(generated_reshape.astype(np.uint8))
            save_img.save('reconstructed_image_test.jpg')

    # make table
    hidden_layer_df = pd.DataFrame(psnr_list, index=hidden_layer_list, columns=neuron_list)

    # generate heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(hidden_layer_df, cmap='GnBu', xticklabels=True, yticklabels=True)
    plt.xlabel('neurons')
    plt.ylabel('layers')
    plt.title('heatmap')
    plt.show()

main()