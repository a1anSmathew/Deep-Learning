import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
from torchvision.models import ResNet18_Weights
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
    transforms.Resize(224),
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
model = models.resnet18(weights = ResNet18_Weights.DEFAULT)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# print(model.fc) # Return the last fc of the CNN model
# print (model.fc.in_features)
for layer in model.children():# model.children() returns an iterator over the immediate layers (modules) inside the model.
    print(layer)
layers = list(model.children())
print(len(layers)) # number of top-level modules
feature_extractor = nn.Sequential(*list(model.children())[:-1])  #[:-1]:- Normal Python list slicing: this removes the last element. * unpacks the list elements when passing to a function
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Passing the training dataset to fine tune the weights
# model.train()
# for X, y in train_dataloader:
#     X, y = X.to(device), y.to(device)  # Moves both the input images and labels to the selected device (cuda if GPU available, else cpu). Ensures computations happen on the same device as the model.
#
#     # Compute prediction error
#     pred = model(X)
#     loss = criterion(pred, y)
#
#     # Backpropagation
#     loss.backward()  # Performs back propagation by calculating gradients for all the parameters which has requires_grad=True.
#     optimizer.step()  # Updates the parameters using the gradients
#     optimizer.zero_grad()  # By default, PyTorch accumulates gradients in .grad (instead of overwriting them). But in standard training, we want fresh gradients each batch, so we reset them to zero.


# Passing the testing dataset to obtain the features
def feature_extraction(dataloader,feature_extractor):
    feature_extractor.eval()
    with torch.no_grad():
        feature_list = []
        labels = []
        for X, y in dataloader :
            X, y = X.to(device), y.to(device)
            pred = feature_extractor(X)          # shape: [1, 512, 1, 1]
            features = pred.view(pred.size(0), -1).cpu().numpy() # flatten -> shape: [512]
            # print(features.shape)                # torch.Size([512])
            feature_list.append(features)
            labels.extend(y.cpu().numpy())
            # print(features[:10])                 # first 10 values for example
        print(len(feature_list))
        feature_list = np.vstack(feature_list)
        labels = np.array(labels)
        return feature_list, labels


# Building an ML Model

X_train,y_train = feature_extraction(train_dataloader,feature_extractor)
X_test, y_test = feature_extraction(test_dataloader,feature_extractor)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", C=1.0, decision_function_shape="ovr")
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("SVM Test Accuracy:", acc)

