import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import os
import pandas as pd
# from torchvision.io import read_image
from PIL import Image

# dataset

def read_image(file_path):
    img = Image.open(file_path)
    return img

class CatAndDog(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.annotations = pd.read_csv(csv_file, names=['image', 'labels'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # The number of sample dataset
    def __len__(self):
        return len(self.annotations)
    
    # Call and return samples based on idx
    def __getitem__(self, idx):

        #iloc for row = idx, col = 0
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    

# test

csv_file = '/Users/yerin/Desktop/research/study/cat_and_dog/cat_dog_100.csv'
img_dir = './cat_dog'
dataset = CatAndDog(csv_file, img_dir)
print("Dataset size = ", len(dataset))

