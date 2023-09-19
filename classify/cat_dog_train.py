import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from cat_and_dog_dataset import CatAndDog
from torchvision.transforms import ToTensor, Lambda, RandomResizedCrop
from torchvision import transforms
import os
from img_normalize import mean, std

device = (
    "cuda"
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

print(f'Using {device} device')


# Hyperparam
num_classes = 2
learning_rate = 1e-3
batch_size = 64
num_epochs = 10

# same image size
# transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#    ToTensor(),
# #    transforms.Normalize([0.4883, 0.4551, 0.4170], [0.2256, 0.2211, 0.2213], inplace=False)
#     transforms.Normalize(mean, std)
# ])

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     ToTensor()
# ])

transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,
                              std, inplace=False)
])

# Load Data
# dataset = CatAndDog('./cat_dog.csv', './cat_dog')
dataset = CatAndDog('/Users/yerin/Desktop/research/study/cat_and_dog/cat_dog_100.csv', '/Users/yerin/Desktop/research/study/cat_and_dog/cat_dog/image')
dataset.transform = transform
dataset.target_transform = Lambda(lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))

# train_set = 80%, test_set = 20% (20000, 5000)
train_set, test_set = torch.utils.data.random_split(dataset, [100, 20])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# test
print('checking dataset')

for i in range(5):
    image, label = dataset[i]
    print(f'Image {i} shape: {image.shape}, Label: {label}')

print('\nchecking train loader')

for images, labels in train_loader:
    print(f'Batch image shape: {images.shape}. Batch labels shape: {labels.shape}')
    break

print('\nchecking test loader')

for images, labels in test_loader:
    print(f'Batch image shape: {images.shape}. Batch labels shape: {labels.shape}')
    break



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        #drop out
        self.dropout = nn.Dropout(p=0.5)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 224 * 224, 10000),
            nn.ReLU(),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2),
        )
        # print('checking Init_NeuralNetwork')

    def forward(self, x):
        x = torch.tanh(x)
        x = self.flatten(x)
        #x = self.dropout(x)
        logits = self.linear_relu_stack(x)

        # logit = class score tensor
        # print('checking Forward_NeuralNetwork')
        return logits
        
# model
model = NeuralNetwork().to(device)
print(model)

# calc loss 
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

print('calc loss')

# Update the weight
optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
#optimizer = optim.Adam(model.parameters(), lr=learning_rate) 


# training model
for epoch in range(num_epochs):

    # update and store into device
    for images, labels in train_loader:
        images = images.to(device)  
        label = labels.to(device)  

        # forward
        class_score = model(images)
        loss = criterion(class_score, labels.float())

        # bakcward for calculating gradient and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    # test
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    # evaluation mode
    model.eval()

    # forward to check accuracy
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            score = model(x)

            prediction = score.argmax(1)
            num_correct += (prediction == y.argmax(1)).sum()
            num_samples += prediction.size(0)

        print(f'got {num_correct} / {num_samples} with accracy {float(num_correct)/float(num_samples) * 100}')

    # training mode
    model.train()

# test

print('checking accuracy on Training set')
check_accuracy(train_loader, model)

print('checking accuracy on Test set')
check_accuracy(test_loader, model)





# Before dropout:
# Epoch [1/10], Loss: 0.22190164029598236
# Epoch [2/10], Loss: 0.23207783699035645
# Epoch [3/10], Loss: 0.22813564538955688
# Epoch [4/10], Loss: 0.22941821813583374
# Epoch [5/10], Loss: 0.2412240207195282
# Epoch [6/10], Loss: 0.20721974968910217
# Epoch [7/10], Loss: 0.1860806941986084
# Epoch [8/10], Loss: 0.2268078774213791
# Epoch [9/10], Loss: 0.24254661798477173
# Epoch [10/10], Loss: 0.22426734864711761

# dropout:
# Epoch [1/2], Loss: 0.2997291088104248
# Epoch [2/2], Loss: 0.2231326699256897
# checking accuracy on Training set
# got 11444 / 20000 with accracy 57.220000000000006
# checking accuracy on Test set
# got 2845 / 5000 with accracy 56.89999999999999

# after reducing the number of dataset and normalizing
# Epoch [1/10], Loss: 0.48068010807037354
# Epoch [2/10], Loss: 0.4503237009048462
# Epoch [3/10], Loss: 0.3539348840713501
# Epoch [5/10], Loss: 0.21571604907512665
# Epoch [6/10], Loss: 0.21422436833381653
# Epoch [7/10], Loss: 0.22367256879806519
# Epoch [8/10], Loss: 0.18784494698047638
# Epoch [9/10], Loss: 0.18945926427841187
# Epoch [10/10], Loss: 0.17549099028110504

# checking accuracy on Training set
# got 97 / 100 with accracy 97.0

# checking accuracy on Test set
# got 12 / 20 with accracy 60.0
