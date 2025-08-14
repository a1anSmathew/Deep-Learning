import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# We want to be able to train our model on an accelerator such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# torch.accelerator.is_available() :- To check if accelerator like GPU is available
# torch.accelerator.current_accelerator().type :- Supposed to return the type of the current accelerator, e.g., "cuda" for GPU or "mps" for Apple Silicon.

