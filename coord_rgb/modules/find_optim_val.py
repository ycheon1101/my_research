from fourier import GaussianFourier
from fourier_mlp import MLP
from img_dataset import img_flatten, xy_flatten, crop_size
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from fourier import GaussianFourier
from PIL import Image
import pandas as pd
from itertools import product

# make table
# num_epochs_list = [200, 300, 400, 500]
num_epochs_list = list(range(200, 501))[::100]

# learning_rate_list = [0.01, 0.001, 0.005]
learning_rate_list = [0.01, 0.001, 0.005]

# scale_list = [1, 2, 3,,,10]
scale_list = list(range(1, 11))

# hidden_feature_list = [32, 64, 128, 256]
hidden_feature_list = list(map(lambda x: 32 * x,[2 ** i for i in range(4)] ))

# hidden_layers_list = [6, 8, 10, 12]
hidden_layers_list = list(range(6, 13))[::2]

list_psnr_fourier = []

# initialize psnr_list
# psnr_list = [np.nan]

# all possible values
all_combinations = list(product(num_epochs_list, learning_rate_list, scale_list, hidden_feature_list, hidden_layers_list))

# [0, 1919]
hyperparam_df = pd.DataFrame(all_combinations, columns=['num_epochs', 'learning_rate', 'scale', 'hidden_feature', 'hidden_layers'])
# psnr -> str
# hyperparam_df['PSNR'] = hyperparam_df['PSNR'].astype(str)

# print(hyperparam_df)

# assign hyper params and calc psnr
for idx, row in hyperparam_df.iterrows():
    num_epochs = int(row['num_epochs'])
    learning_rate = row['learning_rate']
    scale = int(row['scale'])
    hidden_feature = int(row['hidden_feature'])
    hidden_layers = int(row['hidden_layers'])
    # print(num_epochs, learning_rate, scale, hidden_feature , hidden_layers)

    # set the target
    target = img_flatten

    # fourier (mapping size = in_feature / 2)
    fourier_result = GaussianFourier(num_input_channels=2, mapping_size = 128, scale=scale)(xy_flatten)

    # calc loss
    criterion = nn.MSELoss() 

    # Model MLP instanciation
    net = MLP(in_feature=256, hidden_feature=hidden_feature, hidden_layers=hidden_layers, out_feature=3)
    model = net

    # list for storing MSE losses
    losses = []

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        # for mlp
        generated = model(fourier_result)

        loss = criterion(generated, img_flatten)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
    #psnr
    max_pixel = 1.0
    psnr_fourier = [10 * np.log10(max_pixel ** 2 / mse) for mse in losses]

    # hyperparam_df.at[idx, 'PSNR'] = ','.join(map(str, psnr_fourier))
    # hyperparam_df.at[idx, 'PSNR'] = list(map(float, psnr_fourier))

    # make a list for psnr_fourier
    # result_psnr = []
    
    list_psnr_fourier.append(psnr_fourier)
    # result_psnr.append(list_psnr_fourier)
    print(list_psnr_fourier)
    print(f'len_list: {len(list_psnr_fourier)}')

    # print(row['PSNR'])

    # print(row)


# print(hyperparam_df)


# print(hyperparam_df)

# # params
# learning_rate = 5e-3
# num_epochs = 300
# scale = 4

# # set the target
# target = img_flatten

# # fourier (mapping size = in_feature / 2)
# fourier_result = GaussianFourier(num_input_channels=2, mapping_size = 128, scale=scale)(xy_flatten)


# # calc loss
# criterion = nn.MSELoss() 

# # Model MLP instanciation
# net = MLP(in_feature=256, hidden_feature=128, hidden_layers=8, out_feature=3)

# model = net

# # list for storing MSE losses
# losses = []


# # optimizer
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# for epoch in range(num_epochs):

#     # for mlp
#     generated = model(fourier_result)

#     loss = criterion(generated, img_flatten)


#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     losses.append(loss.item())

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# # reshape with final generated
# generated_reshape = model(fourier_result) 
# generated_reshape = torch.reshape(generated_reshape, (crop_size, crop_size, 3))


# # for save image
# # generated_reshape = generated_reshape * 255.
# generated_reshape = generated_reshape.detach().numpy()


# # show image
# plt.imshow(generated_reshape)
# plt.show()

# # save image
# # image_store = Image.fromarray(generated_reshape.astype(np.uint8))
# # image_store.save('scale_4_hidden_128.jpg')

# # psnr
# # max_pixel = 1.0
# # psnr_fourier = [10 * np.log10(max_pixel ** 2 / mse) for mse in losses]


# # plt.figure(figsize=(10, 5))
# # plt.plot(psnr_fourier, label="PSNR_fourier")
# # plt.xlabel("Epochs")
# # plt.ylabel("PSNR")
# # plt.legend()
# # plt.grid(True)
# # plt.title("PSNR")
# # plt.show()





