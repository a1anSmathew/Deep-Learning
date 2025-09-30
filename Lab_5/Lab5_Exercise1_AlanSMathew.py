import numpy as np

# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Step activation
def step(z):
    return np.where(z > 0, 1, 0)

# Hidden layer: 2 neurons
W1 = np.array([[1, 1],
               [1, 1]])
b1 = np.array([0, -1])

# Output layer: 1 neuron
W2 = np.array([[1], [-2]])
b2 = np.array([0])

# Forward pass
z1 = np.dot(X, W1) + b1   # hidden linear
a1 = step(z1)             # hidden step
z2 = np.dot(a1, W2) + b2  # output linear
y_hat = step(z2)          # output step

print("Inputs:\n", X)
print("Predicted XOR:\n", y_hat)
print("True XOR:\n", y)
