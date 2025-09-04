import math

import numpy as np
import pandas as pd
from cffi.ffiplatform import flatten


def conv(A,F,s,p):
    m,n = A.shape
    x,y = F.shape
    A_new = [np.sum(A[i:i+x,j:j+y]*F) for i in range(0,m-x+1,s) for j in range(0,n-y+1,s)]
    A_new = np.array(A_new).reshape((m - x )//s +1, (n - y)// s + 1 )
    A_new = pd.DataFrame(A_new)
    print("\nConvolution Operation Results\n",A_new)

def maxpool(A,F,s,p):
    m, n = A.shape
    x, y = F.shape
    A_new = [np.max(A[i:i + x, j:j + y] * F) for i in range(0,m - x + 1,s) for j in range(0,n - y + 1,s)]
    A_new = np.array(A_new).reshape((m - x + 2 * p)//s +1, (n - y + 2 * p)//s +1 )
    A_new = pd.DataFrame(A_new)
    print("\nMax Pooling Results\n",A_new)

def main():
    # A = np.array([[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]])
    # F = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    # Filter 2x2
    F = np.array([[1, 0],
                  [0, -1]])
    m,n = A.shape
    x,y = F.shape
    s = 1 #Stride
    p = 0
    ans = input("Do you want same convolution? yes = 'y' no = 'n' :- ")
    if ans == 'y':
        p = math.ceil(((m - 1) * s + x - m) / 2)
        A = np.pad(A, ((p, p), (p, p)), mode='constant', constant_values=0)
    print(p)
    print(A)
    conv(A,F,s,p)
    # maxpool(A,F,s,p)

if __name__ == '__main__':
    main()