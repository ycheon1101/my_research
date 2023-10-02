import sys
sys.path.append('/Users/yerin/Desktop/my_research/src/modules')
# return image data frame and crop_size
import table_images
from mlp import MLP
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

# params
learning_rate = 5e-3
num_epochs = 300

# calc loss
criterion = nn.MSELoss()

# instanciation model
model = MLP(in_feature=2, hidden_feature=32, hidden_layers=8, out_feature=3)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# set target
img_df, crop_size = table_images.make_table()
for image in range(1, len(img_df.index) + 1):
    target = img_df['img_flatten'][image]
    xy_flatten = img_df['xy_flatten'][image]

    for epoch in range(num_epochs):

        # generate
        generated = model(xy_flatten)

        # loss = criterion(generated, target)
        loss = criterion(generated, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # reshape the generated tensor to [h, w, c]
    generated_reshape = model(xy_flatten) 
    generated_reshape = torch.reshape(generated_reshape, (crop_size, crop_size, 3))

    # change to numpy for plot image
    generated_reshape = generated_reshape.detach().numpy()

    # show image
    plt.imshow(generated_reshape)
    plt.show()

        
        