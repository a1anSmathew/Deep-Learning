import urllib.request

from torch import nn
from torch.utils.data import DataLoader
from torch.xpu import device
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch import optim
import torch

import numpy as np


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

def loading_data():
    training_data = datasets.CIFAR10(
        root="data2",
        train=True,
        download=True,
        transform=transform
    )
    testing_data = datasets.CIFAR10(
        root="data2",
        train=False,
        download=True,
        transform=transform
    )

    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using {device} device ")

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) # New

    return train_dataloader, test_dataloader, model, criterion

def model_optimization(model, unfreeze_from=-5):
    final_model = nn.Sequential(*list(model.children())[:-1],
                                nn.Flatten(),
                                nn.Linear(512, 10))

    # First freeze everything

    for param in final_model.parameters():
        param.requires_grad = False

    # Unfreeze only the last few layers
    for layer in list(final_model.children())[unfreeze_from:]:
        for param in layer.parameters():
            param.requires_grad = True

    return final_model

def model_training(train_dataloader,model,criterion,optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        loss.backward()  # Performs back propagation by calculating gradients for all the parameters which has requires_grad=True.
        optimizer.step()  # Updates the parameters using the gradients
        optimizer.zero_grad()  # By default, PyTorch accumulates gradients in .grad (instead of overwriting them). But in standard training, we want fresh gradients each batch, so we reset them to zero.

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def model_testing(test_dataloader,model,criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # No gradients computation
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            test_loss += loss.item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
        return correct, test_loss

def epochs(train_dataloader,test_dataloader,model,criterion,optimizer):
    e = 5
    for t in range (e):
        print(f"Epoch {t + 1}\n-------------------------------")
        model_training(train_dataloader,model,criterion,optimizer)
        model_testing(test_dataloader,model,criterion)
    print("Done!")
    torch.save(model.state_dict(), "model4.pth")
    print("Saved PyTorch Model State to model4.pth")


def main():
    train_dataloader, test_dataloader, model, criterion = loading_data()
    print(len(train_dataloader))
    model = model_optimization(model)
    print(list(model.children()))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    epochs(train_dataloader,test_dataloader,model,criterion,optimizer)

if __name__ == '__main__':
    main()