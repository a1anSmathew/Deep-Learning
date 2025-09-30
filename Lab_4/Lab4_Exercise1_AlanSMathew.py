import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])   # shape (4,3)
y = np.array([[0], [1], [1], [0]])  # shape (4,1)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss
def compute_loss(y, y_hat):
    m = y.shape[0]
    return -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))

# Initialize parameters
np.random.seed(42)
W = np.random.randn(3, 1) * 0.01
b = 0
lr = 0.1
iterations = 1000

losses = []

# Training loop
for i in range(iterations):
    # Forward pass
    z = np.dot(X, W) + b
    y_hat = sigmoid(z)

    # Compute loss
    loss = compute_loss(y, y_hat)
    losses.append(loss)

    # Backward pass
    dz = y_hat - y
    dW = np.dot(X.T, dz) / X.shape[0]
    db = np.mean(dz)

    # Update weights
    W -= lr * dW
    b -= lr * db

    # Print loss occasionally
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss:.4f}")

# Final parameters
print("Trained weights:", W.ravel())
print("Trained bias:", b)

# Plot loss curve
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over 1000 Iterations")
plt.show()
