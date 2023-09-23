# import torch
# import imageio.v2 as imageio
# from PIL import Image
# import torchvision.transforms as transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from torch.utils.data import Dataset
# import itertools
# import math
# import torch.optim as optim
# import torch.nn.functional as F


# ####### Create a small MLP ##########
# # class MLP(torch.nn.Module):
# #     def __init__(
# #         self,
# #         in_features: int,
# #         hidden_features: int,
# #         hidden_layers: int,
# #         out_features: int
# #     ):
# #         super().__init__()
# #         out_features = out_features if out_features is not None else hidden_features
# #         self.net = torch.nn.Sequential(
# #             torch.nn.Linear(in_features, hidden_features),
# #             torch.nn.ReLU(),
# #             *[torch.nn.Sequential(
# #                 torch.nn.Linear(hidden_features, hidden_features),
# #                 torch.nn.ReLU()
# #             ) for _ in range(hidden_layers)],
# #             torch.nn.Linear(hidden_features, out_features)
# #         )
# #     def forward(self, x: torch.Tensor):
# #         return self.net(x)



# ################### MAIN ###############

# def main():

#     #0. params 
#     crop_size = 256
#     nb_epochs = 100

#     #1. read the image
#     im = imageio.imread('ca.jpg')

#     #2. convert to tensor
#     im_tensor = torch.Tensor(im).permute(2, 0, 1)
#     my_crop = transforms.CenterCrop(crop_size)
#     im_crop = my_crop(im_tensor)

#     # 3. flatten tensor to [crop_size**2, 3]
#     im_flat = torch.reshape(im_crop,(crop_size*crop_size, 3))

#     #4. Create the mesh grid
#     xy_range = list(map(lambda x: x / crop_size, range(0, crop_size)))
#     xy_range_tensor = torch.tensor(xy_range, dtype=torch.float32)
#     x_grid, y_grid = torch.meshgrid(xy_range_tensor, xy_range_tensor)
#     xy_coord_tensor = torch.stack((x_grid, y_grid), dim=-1)
#     xy_flat = torch.reshape(xy_coord_tensor,(crop_size*crop_size, 2))

#     # 5. Declare network
#     net = MLP(in_features=2, hidden_features=256, hidden_layers=2, out_features=3)

#     # 6. Try one forward
#     #recons_im = torch.reshape(net(xy_flat), (crop_size, crop_size,3))
#     # loss = net(xy_flat) - im_flat


#     ########## TRAIN!
#     # for epoch in range(nb_epochs):
#     #     XXXX = 1 

# main()

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
import itertools
import math
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import imageio.v2 as imageio


####### Create a small MLP ##########
class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int
    ):
        super().__init__()
        out_features = out_features if out_features is not None else hidden_features
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




################### MAIN ###############

def main():

    #0. params 
    crop_size = 400
    nb_epochs = 200
    learning_rate = 5e-3

    #1. read the image
    im = Image.open('ca.jpg')
    im = np.array(im)

    #2. convert to tensor
    im_tensor = torch.Tensor(im).permute(2, 0, 1)
    my_crop = transforms.CenterCrop(crop_size)
    im_crop = my_crop(im_tensor)

    #3. flatten it
    im_permuted = im_crop.permute(1, 2, 0) # Change to [256, 256, 3]
    # im_flat = im_permuted.reshape(-1, 3)  # Flatten and keep the 3 RGB channels
    im_flat = im_permuted.reshape((crop_size * crop_size, 3))  # Flatten and keep the 3 RGB channels
    im_flat = im_flat / 255.0  # Normalizing each RGB value

    # im_crop_flat = im_crop.reshape(3, -1).permute(1, 0)
    # img_view = im_flat.view(crop_size, crop_size, 3) 

    # plt.imshow(img_view)
    # plt.show()




    # im_flat_check = im_crop.reshape(3, -1) / 255.
    # im_flat_check = im_flat_check.permute(1, 0)
    # print(f'im_flat: {im_flat.shape}', f'imflat_check = {im_flat_check.shape}' ,sep='\n')
    
    # print('same') if np.array_equal(im_flat, im_flat_check) else print('not')
    
    #4. Create the mesh grid
    xy_range = list(map(lambda x: x / crop_size, range(0, crop_size)))
    xy_range_tensor = torch.tensor(xy_range, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(xy_range_tensor, xy_range_tensor)
    xy_coord_tensor = torch.stack((x_grid, y_grid), dim=-1)
    xy_flat = torch.reshape(xy_coord_tensor, (crop_size * crop_size, 2))

    # 5. Declare network
    net = MLP(in_features=2, hidden_features=32, hidden_layers=8, out_features=3)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # TRAIN!
    for epoch in range(nb_epochs):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(xy_flat)
        
        # Loss computation
        loss = criterion(outputs, im_flat)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{nb_epochs}, Loss: {loss.item()}")

    # Getting the output from the network and reshaping it
    recons_im = net(xy_flat)  # Assuming the network output is [256*256, 3]
    recons_im = recons_im.view(crop_size, crop_size, 3)  # Reshaping to [256, 256, 3]
    recons_im = recons_im.permute(2, 0, 1)  # Changing to [3, 256, 256] if needed

    # Denormalizing the pixel values
    recons_im = recons_im * 255.0
    recons_im = recons_im.detach().permute(1, 2, 0).numpy()  # Change to [256, 256, 3] for displaying and saving

    print(recons_im)
    recons_im_pil = Image.fromarray(recons_im.astype(np.uint8))
    recons_im_pil.save('reconstructed_image_5.jpg')



if __name__ == "__main__":
    main()
    
    

