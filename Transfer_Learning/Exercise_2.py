import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import DataLoader
import torchvision.models as models
import urllib.request


# Creating a custom dataset
# class CIFAR_Dataset(Dataset):
#     def __init__(self,batch_files,transform = None): #batch_files = path to the different batches of the training data
#         self.data = [] # It will hold all the images
#         self.labels = [] # It will hold all the labels
#         self.transform = transform
#
#         for file in batch_files:
#             with open(file, 'rb') as fo: # Opens in binary read mode
#                 batch = pickle.load(fo, encoding='bytes') # Unpickles the files to objects
#                 print(batch.keys()) # b = bytes object
#                 # print(batch[b'batch_label'])
#                 self.data.append(batch[b'data']) # this add a list of 10,000 * 3072 to the self.data each time
#                 self.labels += batch[b'labels'] #
#
#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # N x C x H x W, Each row is converted into a 3*32*32
#         self.data = self.data.transpose((0, 2, 3, 1))  # N x H x W x C (PIL image format expects this (H*W*C))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         img, label = self.data[idx], self.labels[idx]
#         from PIL import Image
#         img = Image.fromarray(img) # Converts the 32*32*3 numerical values into proper image
#
#         if self.transform: # If we are transforming the image then we can do it
#             img = self.transform(img)
#
#         return img, label
#
# transform = transforms.Compose([ # .Compose helps us to chain multiple Transform operations
#     transforms.ToTensor() # Here it is only ToTensor
# ])
#
# train_path = [
#     "/home/ibab/Deep_Learning/TransferLearning_ResNet/data/data_batch_1",
#     "/home/ibab/Deep_Learning/TransferLearning_ResNet/data/data_batch_2",
#     "/home/ibab/Deep_Learning/TransferLearning_ResNet/data/data_batch_3",
#     "/home/ibab/Deep_Learning/TransferLearning_ResNet/data/data_batch_4",
#     "/home/ibab/Deep_Learning/TransferLearning_ResNet/data/data_batch_5"
# ]
#
# trainset = CIFAR_Dataset(train_path, transform=transform)
# #print(trainset.data.shape)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

proxy = "http://245hsbd001%40ibab.ac.in:7619399357Alan@proxy.ibab.ac.in:3128"

proxy_handler = urllib.request.ProxyHandler({
    "http": proxy,
    "https": proxy,
})
opener = urllib.request.build_opener(proxy_handler)
urllib.request.install_opener(opener)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

training_data = datasets.CIFAR10( #Training_data will hold the training dataset and FashionMNIST is a preset dataset
    root="data2", #This tells the loader to store or look for the dataset in the folder named "data". If the folder doesn't exist, it will be created automatically.
    train=True,
    download=True,
    transform=transform #Converts images into Tensors
)

test_data = datasets.CIFAR10(
    root="data2",
    train=False,
    download=True,
    transform=transform
)

batch_size = 64
device = "cpu"
print(f"Using {device} device")

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# # Building a CNN Model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         # Conv Layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)# 1x28x28 → 32x28x28 (No. of Channels, Number of filters, kernel_size, strides, padding)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)# 32x28x28 → 64x28x28
#         self.pool = nn.MaxPool2d(2, 2) # Halves size → 64x14x14 (kernel_size,stride)
#         self.ReLU = nn.ReLU()
#         self.flatten = nn.Flatten()
#
#         # Fully Connected Layers
#         self.fc1 = nn.Linear(64*16*16,128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self,x):
#         # print(x.shape)
#         x = self.ReLU(self.conv1(x))
#         # print(x.shape)
#         x = self.ReLU(self.conv2(x))
#         # print(x.shape)
#         x = self.pool(x)
#         # print(x.shape)
#         x = self.flatten(x)
#         x = self.ReLU(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = CNN().to(device)
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1],
                                nn.Flatten(),
                                nn.Linear(512,10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the data
def Train(train_dataloader,model,criterion,optimizer):
    size = len(train_dataloader.dataset) # Length/Number of samples
    model.train() # Sets the model in training mode
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device) # Moves both the input images and labels to the selected device (cuda if GPU available, else cpu). Ensures computations happen on the same device as the model.

        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        loss.backward() # Performs back propagation by calculating gradients for all the parameters which has requires_grad=True.
        optimizer.step() # Updates the parameters using the gradients
        optimizer.zero_grad() # By default, PyTorch accumulates gradients in .grad (instead of overwriting them). But in standard training, we want fresh gradients each batch, so we reset them to zero.

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def Test(test_dataloader, model, criterion):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # No gradients computation
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item() # .item = Converts tensor to a float
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # pred.argmax(1) = picks the index of the max logit per row
            # .type(torch.float) = converts true/false to float numbers
            # .sum().item() = sums up everything and converts to a python float
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    Train(train_dataloader, model, criterion, optimizer)
    Test(test_dataloader, model, criterion)
print("Done!")
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
