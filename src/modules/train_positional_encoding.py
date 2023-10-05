import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from positional_encoding import GaussianFourier
import table_images
from mlp import MLP

# params
learning_rate = 5e-3
num_epochs = 300
hidden_feature = 128
hidden_layers = 8
max_pixel = 1.0

# calc loss
criterion = nn.MSELoss() 

# Model MLP instanciation
model = MLP(in_feature=256, hidden_feature=hidden_feature, hidden_layers=hidden_layers, out_feature=3)

# get img_data
img_df, crop_size = table_images.make_table()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# train model
def train_model(target, model_arg):
    for epoch in range(num_epochs):

        # model
        generated = model(model_arg)

        # loss = criterion(generated, target)
        loss = criterion(generated, target)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# plot image
def plot_img(model_arg):
    # reshape with final generated
    generated_reshape = model(model_arg)
    generated_reshape = torch.reshape(generated_reshape, (crop_size, crop_size, 3))

    generated_reshape = generated_reshape.detach().numpy()


    # show image
    plt.imshow(generated_reshape)
    plt.show()

# main
def main():
    for image in range(1, len(img_df.index) + 1):
        target = img_df['img_flatten'][image]
        xy_flatten = img_df['xy_flatten'][image]
        fourier_result = GaussianFourier(num_input_channels=2, mapping_size = 128, scale=4)(xy_flatten)

        train_model(target, fourier_result)
        plot_img(fourier_result)

# main()






