import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3087,))
])

test_data = datasets.MNIST(
    root="data",
    download=True,
    train = False,
    transform=transform
)

train_data = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=transform
)

batch_size = 64
device = "cpu"

test_dataloader = DataLoader(test_data,batch_size=batch_size)
train_dataloader = DataLoader(train_data,batch_size=batch_size)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,3,3,1,1)
        self.fc1 = nn.Linear(3*28*28,10)
        self.flatten = nn.Flatten()
        self.ReLu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.ReLu(self.conv1(x))
        x = self.flatten(x)
        x = self.softmax(self.fc1(x))
        # print(x.shape)
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimz = optim.Adam(model.parameters(), lr = 0.001)

# Training

def Train(train_dataloader, optimz,criterion,model):
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(train_dataloader):
        X,y = X.to(device) , y.to(device)
        pred = model(X)
        loss = criterion(pred,y)

        loss.backward()
        optimz.step()
        optimz.zero_grad()

        if batch % 100 == 0:
            loss,current = loss.item(), (batch+1) * len(X)
            print(f"Batch {batch}: Loss {loss}")
            # print(f"loss:{loss:>7f} [{current:>5d} / {size:>5d}]")

def Test(test_dataloader,criterion,model):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss , correct = 0,0
    with torch.no_grad():
        for X,y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            test_loss += loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"test loss:{test_loss} , accuracy:{100*correct}")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    Train(train_dataloader, optimz,criterion,model)
    Test(test_dataloader,criterion,model)
print("Done!")

# for x,y in train_dataloader:
#     pred = model(x)
#     print(pred.shape)
#     break