from img_dataset import img_flatten, xy_flatten, crop_size
from model import model
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

# params
learning_rate = 0.01
num_epochs = 701

# set the target
# target = img_flatten
target = img_flatten

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):

    model.train()

    # for mlp
    generated = model(xy_flatten)
    # generated = generated.permute(2, 0, 1)
    # print(f'generated_shape: {generated.shape}')

    # loss = criterion(generated, target)
    loss = nn.functional.l1_loss(target, generated)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
        # print(generated.shape)
        # generated = generated.view((crop_size, crop_size, 3)).detach().numpy()
        # print(generated.shape)
        generated = torch.reshape(generated, (crop_size, crop_size, 3))
        # generated = torch.reshape(generated, (crop_size, crop_size, 3)).detach().numpy()
        # generated = generated * 255
        # print(f'generated: {generated}')
        # generated = (generated * 255).astype(np.uint8)
        # generated = (generated - generated.min()) / (generated.max() - generated.min())
        # print(generated)
        # generated = generated.permute(1, 2, 0).detach().numpy()
        generated = generated.detach().numpy()
        plt.imshow(generated)
        plt.show()
        # plt.pause(10)
        # plt.close()
model.eval()
