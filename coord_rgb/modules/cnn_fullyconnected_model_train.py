import torch.nn as nn
from img_dataset import img_cropped, xy_coord_tensor
from cnn_model import model
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from img_dataset import img_cropped
from img_dataset_cnn import xy_coord_tensor


class CNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        self.neural_cnn_net = nn.Sequential(
            
            nn.Conv2d(
                in_features,
                hidden_features,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features),

            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features),

            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features),

            nn.Conv2d(
                hidden_features,
                100,
                kernel_size=1,
                padding=0),
        )
        self.fullyconnected = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            
            nn.Linear(100, out_features),
            nn.ReLU(),
            nn.Dropout(p = 0.5),

        )

    def forward(self, x):
        x = self.neural_cnn_net(x)
        # x = x.squeeze(0).permute(1, 2, 0)
        x = x.squeeze(0)
        print(f'x: {x.shape}')
        y = self.fullyconnected(x)
        print(f'y: {y.shape}')
        return self.fullyconnected(x)

# declare model
net = CNN(in_features=2, hidden_features=300, out_features=3)

model = net

# params
learning_rate = 0.01
num_epochs = 701

# set the target
# target = img_flatten
img_cropped = img_cropped.unsqueeze(0)

target = img_cropped
print(f'target.shape: {target.shape}')
# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):

    model.train()

    # for mlp
    generated = model(xy_coord_tensor)
    # generated = generated.permute(2, 0, 1)
    print(f'generated_shape: {generated.shape}')

    # loss = criterion(generated, target)
    if target.shape != generated.shape:
        target = target.squeeze(0)
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
