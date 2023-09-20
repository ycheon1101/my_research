from img_dataset import img_cropped, xy_coord_tensor
from cnn_model import model
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from img_dataset import img_cropped
from img_dataset_cnn import xy_coord_tensor

# params
learning_rate = 0.01
num_epochs = 701

# set the target
# target = img_flatten
img_cropped = img_cropped.unsqueeze(0)

target = img_cropped

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):

    model.train()

    # for mlp
    generated = model(xy_coord_tensor)
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
        # generated = torch.reshape(generated, (crop_size, crop_size, 3)).detach().numpy()
        # generated = generated * 255
        print(f'generated: {generated}')
        # generated = (generated * 255).astype(np.uint8)
        # generated = (generated - generated.min()) / (generated.max() - generated.min())
        # print(generated)
        generated = generated.squeeze(0).permute(1, 2, 0).detach().numpy()
        plt.imshow(generated)
        plt.show()
        # plt.pause(10)
        # plt.close()
model.eval()
