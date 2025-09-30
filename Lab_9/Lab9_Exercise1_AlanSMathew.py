import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Settings
layers = 50              # number of layers
x = 1.0                  # input
target = 0.5             # dummy target
learning_rate = 0.01

def simulate(weight_init, title):
    W = np.full((layers,), weight_init)

    grads = []
    # Forward pass
    a = x
    activations = [a]
    for i in range(layers):
        z = W[i] * a
        a = z  # no activation, pure linear
        activations.append(a)

    # Loss: simple squared error
    loss = 0.5 * (a - target) ** 2

    # Backward pass
    da = a - target  # dL/da at last layer
    for i in reversed(range(layers)):
        dz = da                 # since activation is identity
        dW = dz * activations[i]
        da = dz * W[i]          # chain rule: pass gradient backwards
        grads.append(abs(dW))

    grads.reverse()  # because we looped backward

    # Plot gradient magnitudes across layers
    plt.plot(range(1, layers+1), grads, marker='o')
    plt.yscale("log")
    plt.xlabel("Layer")
    plt.ylabel("Gradient magnitude (log scale)")
    plt.title(title + f" (init={weight_init})")
    plt.show()

# Case 1: Vanishing gradient (small weight < 1)
simulate(weight_init=0.5, title="Vanishing Gradient")

# Case 2: Exploding gradient (large weight > 1)
simulate(weight_init=1.5, title="Exploding Gradient")
