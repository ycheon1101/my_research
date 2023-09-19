import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor




# img dir 
img_dir = "/Users/yerin/Desktop/research/study/cat_and_dog/cat_dog_class"



# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(img_dir, transform)

# data load
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# initialize
mean = 0.
std = 0.
total_samples = 0


for data in loader:
    images, _ = data
    batch_samples = images.size(0)
    # image width
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_samples += batch_samples

mean /= total_samples
std /= total_samples

print(f'Mean: {mean}')
print(f'Standard Deviation: {std}')