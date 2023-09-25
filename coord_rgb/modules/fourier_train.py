from img_dataset import img_flatten, xy_flatten, crop_size
# from model import model
from fourier_mlp import MLP
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from fourier import GaussianFourier
from PIL import Image
from train import losses_train

# params
learning_rate = 5e-3
num_epochs = 300

# set the target
# target = img_flatten
target = img_flatten

# fourier (mapping size = in_feature / 2)
fourier_result = GaussianFourier(num_input_channels=2, mapping_size = 128, scale=4)(xy_flatten)

net = MLP(in_feature=256, hidden_feature=128, hidden_layers=8, out_feature=3)

model = net

# calc loss
criterion = nn.MSELoss() 

# Model MLP instanciation

# list for storing MSE losses
losses = []


# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):

    # model.train()

    # for mlp
    generated = model(fourier_result)
    # generated = generated.permute(2, 0, 1)
    # print(f'generated_shape: {generated.shape}')

    # loss = criterion(generated, target)
    loss = criterion(generated, img_flatten)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# reshape with final generated
generated_reshape = model(fourier_result) 
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
max_pixel = 1.0
psnr_fourier = [10 * np.log10(max_pixel ** 2 / mse) for mse in losses]
psnr = [10 * np.log10(max_pixel ** 2 / mse) for mse in losses_train]

plt.figure(figsize=(10, 5))
plt.plot(psnr_fourier, label="PSNR_fourier")
plt.plot(psnr, label="PSNR")
plt.xlabel("Epochs")
plt.ylabel("PSNR")
plt.legend()
plt.grid(True)
plt.title("PSNR")
plt.show()



