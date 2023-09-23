from img_dataset import img_flatten, xy_flatten, crop_size
# from model import model
from mlp import model
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
# from PIL import Image

# params
learning_rate = 5e-3
num_epochs = 300

# set the target
# target = img_flatten
target = img_flatten

# calc loss
criterion = nn.MSELoss() 

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):

    # model.train()

    # for mlp
    generated = model(xy_flatten)
    # generated = generated.permute(2, 0, 1)
    # print(f'generated_shape: {generated.shape}')

    # loss = criterion(generated, target)
    loss = criterion(generated, img_flatten)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # if epoch % 100 == 0:
    #     print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
    #     # print(generated.shape)
    #     # generated = generated.view((crop_size, crop_size, 3)).detach().numpy()
    #     # print(generated.shape)
    #     generated = torch.reshape(generated, (crop_size, crop_size, 3))
    #     # generated = torch.reshape(generated, (crop_size, crop_size, 3)).detach().numpy()
    #     # generated = generated * 255
    #     # print(f'generated: {generated}')
    #     # generated = (generated * 255).astype(np.uint8)
    #     # generated = (generated - generated.min()) / (generated.max() - generated.min())
    #     # print(generated)
    #     # generated = generated.permute(1, 2, 0).detach().numpy()
    #     generated = generated.detach().numpy()
    #     plt.imshow(generated)
    #     plt.show()
        # plt.pause(10)
        # plt.close()
# model.eval()

# [160000, 3]
# print(f'generated shape: {generated.shape}')

# generated_reshape = model(xy_flatten) 
generated_reshape = torch.reshape(generated, (crop_size, crop_size, 3))
# generated_reshape = generated_reshape.permute(2, 0, 1) * 255.0

# generated_reshape = generated_reshape.permute(1, 2, 0)



# # print(generated_reshape)


# # print(f'generated_reshape: {generated_reshape.shape}')
generated_reshape = generated_reshape.detach().numpy()
# generated_reshape = generated_reshape.astype(np.uint8)


plt.imshow(generated_reshape)
plt.show()



