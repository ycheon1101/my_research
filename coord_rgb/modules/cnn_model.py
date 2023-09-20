import torch.nn as nn
from img_dataset import crop_size

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
                out_features,
                kernel_size=1,
                padding=0),
            nn.Sigmoid(),

        )

    def forward(self, x):
        return self.neural_cnn_net(x)

# declare model
net = CNN(in_features=2, hidden_features=300, out_features=3)

model = net