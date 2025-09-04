import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.ops.nn_impl import relu_layer
from torch import optim
#x = torch.tensor([-1.0, 0.0, 2.0])

    # Using functional
    #print(F.relu(x))   # tensor([0., 0., 2.])

    # Using nn.Module
    #relu_layer = nn.ReLU()
    #print(relu_layer(x))  # tensor([0., 0., 2.])
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

training_data = datasets.FashionMNIST( #Training_data will hold the training dataset and FashionMNIST is a preset dataset
    root="data", #This tells the loader to store or look for the dataset in the folder named "data". If the folder doesn't exist, it will be created automatically.
    train=True,
    download=True,
    transform=transform #Converts images into Tensors
)

test_data = datasets.FashionMNIST(
    root="data",
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

# Building a CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)# 1x28x28 → 32x28x28 (No. of Channels, Number of filters, kernel_size, strides, padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)# 32x28x28 → 64x28x28
        self.pool = nn.MaxPool2d(2, 2) # Halves size → 64x14x14 (kernel_size,stride)
        self.ReLU = nn.ReLU()
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(64*14*14,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        x = self.ReLU(self.conv1(x))
        x = self.ReLU(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
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
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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
