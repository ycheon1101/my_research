import torch.nn as nn
import torch
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_feature, out_feature, is_first=False):
        super().__init__()

        self.is_first = is_first
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.linear = nn.Linear(in_feature, out_feature)

        self.set_linear()

    def set_linear(self):
        with torch.no_grad():
            if self.is_first:
                weight = torch.empty(self.out_feature, self.in_feature)
                self.linear.weight.uniform_(-1 / self.in_feature, 1 / self.in_feature)
                self.linear.weight = nn.Parameter(weight)
                self.w0 = 1.
            else:
                weight = torch.empty(self.out_feature, self.in_feature)
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_feature), np.sqrt(6 / self.in_feature))
                self.linear.weight = nn.Parameter(weight)
                self.w0 = 30    # 30 in paper

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))



class SIREN_MLP(nn.Module):
    def __init__ (self, in_feature: int, hidden_feature: int, hidden_layers: int, out_feature: int):
        super().__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.hidden_layers = hidden_layers
        self.out_feature = out_feature
        

        # first layer
        layers = [
                SineLayer(in_feature=in_feature, out_feature=hidden_feature, is_first=True)
        ]

        # last layer
        out_layer_list = [
            SineLayer(in_feature=hidden_feature, out_feature=out_feature, is_first=False),
            nn.Sigmoid()
        ]

        # hidden layers
        for _ in range(hidden_layers):
            layers.extend([
                SineLayer(in_feature=hidden_feature, out_feature=hidden_feature, is_first=False)
            ])
        
        layers.extend(out_layer_list)

        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)




