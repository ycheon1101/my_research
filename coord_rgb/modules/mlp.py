import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__ (self, in_feature, hidden_feature, hidden_layers, out_feature):
        super().__init__()

        # first layer
        layers = [
                nn.Linear(in_feature, hidden_feature),
                nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
                # nn.Dropout(p=0.5)
        ]

        # hidden layer
        hidden_layer_list = [
            nn.Linear(hidden_feature, hidden_feature),
            nn.BatchNorm1d(hidden_feature),
            nn.ReLU(),
            # nn.Dropout(p = 0.5)
        ]

        # last layer
        out_layer_list = [
            nn.Linear(hidden_feature, out_feature),
            nn.Sigmoid()
        ]

        for _ in range(hidden_layers):
            layers.extend(hidden_layer_list)
        
        layers.extend(out_layer_list)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP2(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int
    ):
        super().__init__()
        # out_features = out_features if out_features is not None else hidden_features
        layers = [
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.BatchNorm1d(hidden_features),  # Batch Normalization layer
            torch.nn.ReLU(),
        ]
        for _ in range(hidden_layers):
            layers.extend([
                torch.nn.Linear(hidden_features, hidden_features),
                torch.nn.BatchNorm1d(hidden_features),  # Batch Normalization layer
                torch.nn.ReLU(),
            ])
        layers.append(torch.nn.Linear(hidden_features, out_features))
        layers.append(torch.nn.Sigmoid())  # Assuming you want to keep the Sigmoid on the output layer

        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.net(x)

# declare network
# net = MLP(in_feature=2, hidden_feature=32, hidden_layers=8, out_feature=3)
net_2 = MLP2(in_features=2, hidden_features=32, hidden_layers=8, out_features=3)

# print(f'net: {net}')
# print(f'net: {net_2}')


# model = net
model = net_2


