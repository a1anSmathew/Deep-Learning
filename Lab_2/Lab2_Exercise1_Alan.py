import ast
import numpy as np

def ReLu_Activation(x):
    val = np.maximum(0,x)

    return val

def Softmax_Activation(x):
    soft_vec = []
    den = sum([np.exp(j) for j in x])
    for i in x:
        soft = np.exp(i) / (den)
        soft_vec.append(soft)

    soft_vec = np.array(soft_vec)

    return soft_vec

def Neuron_computation(prev,weights,i,j):
    w = 0
    pre = input("Should preset weights be applied ?(y = yes and n = no) ")
    if pre == "y" or pre == "yes":
        w = weights[j]
        w = w[i]
    if pre == "n" or pre == "no":
        w = (input("Enter the value of weights as list Eg. [0.1,0.24,0.66]: "))

    if isinstance(w, str):
        w = ast.literal_eval(w)
    val = np.dot(prev,w)
    bias = float(input("Input the bias of the neuron: "))
    val = np.add(val,bias)

    return val


def Layer_Computation(prev,j,weights):
    layer = []
    n = int(input("Number of neurons in current layer: "))
    for i in range (n):
        val = Neuron_computation(prev,weights,i,j)
        layer.append(val)
    layer = np.array(layer)
    layer = ReLu_Activation(layer)

    return layer

def Output_Computation(prev,k,weights):
    layer = []
    n = int(input("Number of neurons in current layer: "))
    for i in range (n):
        val = Neuron_computation(prev,weights,i,k)
        layer.append(val)
    layer = np.array(layer)
    layer = Softmax_Activation(layer)
    # layer = ReLu_Activation(layer)

    return layer

def Feed_Forward_Neural_Network(input_vector,weights):
    layer_values = []
    k = int(input("Enter the number of layers (including output layer): "))

    prev = input_vector

    # Hidden layers: ReLU activation
    for j in range(k - 1):
        print(f"\nLayer {j + 1} (ReLU):")
        layer = Layer_Computation(prev,j,weights)
        prev = layer
        layer_values.append(prev)

    # Final layer: Softmax activation
    print(f"\nOutput Layer (Softmax):")
    output = Output_Computation(prev,k-1,weights)
    layer_values.append(output)

    print("\nFinal Output (Softmax probabilities):")
    print(output)

def main():
    input_vector = np.array([-2.4, 1.2, -0.8, 1.1])
    W1 = [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    W2 = [[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]]
    W3 = [[0.01, 0.01], [0.01, 0.01]]
    weights = [W1, W2, W3]
    Feed_Forward_Neural_Network(input_vector,weights)


if __name__ == '__main__':
    main()