import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
# target_transform → applies a transformation to the label (y), not the image.
# Lambda(...) → lets you define a small inline function for that transformation.
# y is the class label (an integer 0–9 for FashionMNIST categories).
# torch.zeros(10, dtype=torch.float)
#   Creates a 1D tensor of length 10, filled with 0.0.
#   This will be the one-hot encoding.
#.scatter_(0, torch.tensor(y), value=1)
#   scatter_ (with the underscore) → in-place operation.
#   Places the value 1 at the position/index y along dimension 0.
#   For example, if y=3 → the tensor becomes:
#       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]

