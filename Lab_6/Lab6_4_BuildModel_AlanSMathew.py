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


# Define the Class
class NeuralNetwork(nn.Module): # Create a class called "NeuralNetwork" which inherits from the nn.Module that gives your class all the building blocks for layers, forward passes, parameters, etc.
    def __init__(self): # Constructor
        super().__init__() # runs the parent’s setup logic
        self.flatten = nn.Flatten() # nn.Flatten() reshapes each image into a 1D vector: Example: (1, 28, 28) → (784,).
        self.linear_relu_stack = nn.Sequential( # lets you chain multiple layers in order.
            nn.Linear(28*28, 512), # Fully connected layer: input size = 784 pixels, output = 512 features.
            nn.ReLU(), # Non-linear activation (ReLU = Rectified Linear Unit).
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x): # Forward Pass
        x = self.flatten(x) # Flattens the input images into vectors (so they can enter the linear layers).
        logits = self.linear_relu_stack(x) # Passes the flattened input through the sequence of layers (linear → ReLU → linear → ReLU → linear).
        return logits

model = NeuralNetwork().to(device) # Creates an instance of the class
print(model)

# .to(device) :- Pytorch randomly initializes parameters and buffers.
# Buffers are for example Some layers (like batch normalization or dropout) need extra numbers that are not trained but still important. not updated by backprop, but still part of the model’s state.
# Hey, put all the numbers my model needs (its learnable weights and helper values) onto this chip (CPU or GPU), so it can do calculations there.

# Input to the model
X = torch.rand(1, 28, 28, device=device) # Creates a tensor 28*28 filled with random number
logits = model(X) # This sends the input X through the neural network.
pred_probab = nn.Softmax(dim=1)(logits) # nn.Softmax(dim=1) converts logits into probabilities. dim=1 means: for each row (image), apply softmax across the 10 class scores.
y_pred = pred_probab.argmax(1) # argmax(1) = find the index of the largest value along dimension 1.
print(f"Predicted class: {y_pred}")


input_image = torch.rand(3,28,28) # 3 → this is the batch size (you created 3 images at once).
# input_image = torch.rand(3, 1, 28, 28)  # batch of 3 grayscale images. The input usually is (batch_size, channels, height, width)
print(input_image.size())

# Flattening the image
flatten = nn.Flatten() # PyTorch layer that converts multi-dimensional input into 2D. It keeps the first dimension (batch size) as it is. Then it flattens everything else into one long vector.
flat_image = flatten(input_image)
print(flat_image.size()) # It becomes 3 1D vectors

# Linear Layer
layer1 = nn.Linear(in_features=28*28, out_features=20) #in_features=28*28 = 784 → the input vector length (each image after flattening). out_features=20 → the layer will output a vector of length 20 for each input.
hidden1 = layer1(flat_image)
print(hidden1.size())


# ReLu Layer
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1) # Applying ReLu activation function on hidden layer
print(f"After ReLU: {hidden1}")


#nn.Sequential
seq_modules = nn.Sequential( # It lets you stack layers (or functions) in a sequence, so the input flows through them in order.
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)


# Softmax on last layer
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


# Model Parameters
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")