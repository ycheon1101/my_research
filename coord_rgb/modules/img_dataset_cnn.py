from img_dataset import xy_coord_tensor



xy_coord_tensor = xy_coord_tensor.unsqueeze(0).permute(0, 3, 1, 2)

