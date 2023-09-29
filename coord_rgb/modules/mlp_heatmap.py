
import torch.nn as nn
import torch
# from img_dataset import xy_flatten
# from img_dataset import img_flatten, xy_flatten, crop_size
# from model import model
# from fourier_mlp import MLP
# import torch.optim as optim
# import numpy as np
# from fourier import GaussianFourier



# class MLP(nn.Module):
#         def __init__ (self, in_feature: int, hidden_feature: int, hidden_layers: int, out_feature: int):
#             super().__init__()

#             self.hidden_layers = hidden_layers

#             self.first_layer = nn.Sequential(
#                 nn.Linear(in_feature, hidden_feature),
#                 nn.BatchNorm1d(hidden_feature),
#                 nn.ReLU()
#             )

#             self.hidden_layer = nn.Sequential(
#                 nn.Linear(hidden_feature, hidden_feature),
#                 nn.BatchNorm1d(hidden_feature),
#                 nn.ReLU()
#             )
            
#             self.last_layer = nn.Sequential(
#                 nn.Linear(hidden_feature, out_feature),
#                 nn.Sigmoid()
#             )

#         def forward(self, x):
#             # list for hidden layer features
#             hidden_layer_features = []
            
#             # first layer
#             x = self.first_layer(x)

#             # hidden layer
#             for _ in range(self.hidden_layers):
#                 x = self.hidden_layer(x)
#             hidden_layer_features.append(x)
                
            
#             x = self.last_layer(x)

#             return x, hidden_layer_features
        



        
# import torch.nn as nn
# import torch


# class MLP(nn.Module):
#     def __init__ (self, in_feature: int, hidden_feature: int, hidden_layers: int, out_feature: int):
#         super().__init__()

#         self.hidden_layer = nn.ModuleList([])

#         # first layer
#         self.first_layer = nn.Sequential(
#             nn.Linear(in_feature, hidden_feature),
#             nn.BatchNorm1d(hidden_feature),
#             nn.ReLU()
#         )

        

#         # last layer
#         self.last_layer = nn.Sequential(
#             nn.Linear(hidden_feature, out_feature),
#             nn.Sigmoid()
#         )

#         # hidden layers
#         for _ in range(hidden_layers):
#             self.hidden_layer.extend([
#                 nn.Linear(hidden_feature, hidden_feature),
#                 nn.BatchNorm1d(hidden_feature),
#                 nn.ReLU()
#             ])
        
#         # layers.extend(hidden_layer)
#         # layers.extend(out_layer_list)

#         # self.net = torch.nn.Sequential(*layers)

#     def forward(self, x):

#         hidden_layer_features = []

#         x = self.first_layer(x)

#         for layer in self.hidden_layer:
#             x = layer(x)
#             hidden_layer_features.append(x)

#         # hidden_layer_features = torch.stack(hidden_layer_features)
            

#         x = self.last_layer(x)
        
#         return x, hidden_layer_features


# # declare network
# # net = MLP(in_feature=256, hidden_feature=32, hidden_layers=8, out_feature=3)

# net = MLP(in_feature=256, hidden_feature=128, hidden_layers=8, out_feature=3)




# # print(f'net: {net}')



# # model = net














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
        






