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


def FFNN(input_vector,weights):
    prev = input_vector.reshape(-1,1)
    Layers = []
    # Hidden Layers Computation
    for i in range(len(weights)-1):
        i = np.array(weights[i])
        HL = np.dot(i,prev)
        HL = ReLu_Activation(HL)
        Layers.append(HL)
        prev = HL

    # Output Layer Computation
    i = len(weights)
    output = np.dot(weights[i-1],prev)
    output = Softmax_Activation(output)
    Layers.append(output)
    Layers = [layer.tolist() for layer in Layers]
    print(Layers)



def main():
    input_vector = np.array([-2.4,1.2,-0.8,1.1])
    W1 = [[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]]
    W2 = [[0.001,0.001,0.001],[0.001,0.001,0.001]]
    W3 = [[0.01,0.01],[0.01,0.01]]
    weights = [W1,W2,W3]
    FFNN(input_vector,weights)


if __name__ == '__main__':
    main()