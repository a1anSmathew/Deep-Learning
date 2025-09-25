# def RNN():
import numpy as np

def Reshaping(Wxh,Wyh,Whh,x1,ho,yd):
    x,y = ho.shape
    a,b = x1.shape
    Whh = Whh.reshape(-1,x)
    Wxh = Wxh.reshape(-1,a)
    Wyh = Wyh.reshape(-1,yd)

    return Whh,Wxh,Wyh
def TanH(x):
    e = 2.71
    h = (e**x - e**-x)/(e**x + e**-x)

    return h

def temp(Whh,Whx,h,x):
    a = Whh @ h
    b = Whx @ x
    c = a + b
    h = TanH(c)

    return h

def main():

    # Given Details
    ho = np.array([[0],[0],[0]])
    Wxh = np.array([0.5,-0.3,0.8,0.2,0.1,0.4])
    Wyh = np.array([1,-1,0.5,0.5,0.5,-0.5])
    Whh = np.array([0.1,0.4,0.0,-0.2,0.3,0.2,0.05,-0.1,0.2])
    x1 = np.array([[1],[2]])
    x2 = np.array([[-1],[1]])
    yd = 2

    # Reshaping the Weight Matrices
    Whh,Wxh,Wyh = Reshaping(Wxh,Wyh,Whh,x1,ho,yd)

    # Multiplications
    input = np.array([x1,x2])
    for x in input:
        h = temp(Whh,Wxh,ho,x)
        print("h = ",h)
        ho = h



if __name__ == '__main__':
    main()