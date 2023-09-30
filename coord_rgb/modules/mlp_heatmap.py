
import torch.nn as nn
import torch

class MLP(nn.Module):
        def __init__ (self, in_feature: int, hidden_feature: int, hidden_layers: int, out_feature: int):
            super().__init__()
            
            

            self.first_layer = nn.Sequential(
                nn.Linear(in_feature, hidden_feature),
                nn.BatchNorm1d(hidden_feature),
                nn.ReLU()
            )

            self.hidden_layer_list = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_feature, hidden_feature),
                    nn.BatchNorm1d(hidden_feature),
                    nn.ReLU()
                ) for _ in range(hidden_layers)
            ])
            
            self.last_layer = nn.Sequential(
                nn.Linear(hidden_feature, out_feature),
                nn.Sigmoid()
            )

        def forward(self, x):
            # list for hidden layer features
            hidden_layer_features = []
            # hidden_layer_features = torch.tensor(hidden_layer_features)
            
            # first layer
            x = self.first_layer(x)

            # hidden layer
            for layer in self.hidden_layer_list:
                x = layer(x)
                hidden_layer_features.append(x)

            hidden_layer_features_stack = torch.stack(hidden_layer_features)
                
            
            x = self.last_layer(x)

            return x, hidden_layer_features_stack
        






