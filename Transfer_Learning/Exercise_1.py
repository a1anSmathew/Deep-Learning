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


# Loading the model and removing the last layer
model = models.resnet18(pretrained=True)
# print(model.fc) # Return the last fc of the CNN model
# print (model.fc.in_features)
for layer in model.children():# model.children() returns an iterator over the immediate layers (modules) inside the model.
    print(layer)
layers = list(model.children())
print(len(layers)) # number of top-level modules
feature_extractor = nn.Sequential(*list(model.children())[:-1])  #[:-1]:- Normal Python list slicing: this removes the last element. * unpacks the list elements when passing to a function
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Passing the training dataset to obtain the features
model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = feature_extractor(X)
        print(len(X))