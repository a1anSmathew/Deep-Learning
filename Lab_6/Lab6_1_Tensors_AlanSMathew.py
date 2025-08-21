import torch
import numpy as np

# Directly from Data
data = [[1, 2],[3, 4]] #Matrix
x_data = torch.tensor(data) #Converting it into a Tensor

# From a Numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array) # Converting it into a tensor


x_ones = torch.ones_like(x_data) # Converts all the values of x_data into 1's
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data and changes the datatype of c_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape) # Creates a tensor of shape 2*3 filled with random numbers
ones_tensor = torch.ones(shape) # Creates a tensor of shape 2*3 filled with 1 (float by default)
zeros_tensor = torch.zeros(shape) # Creates a tensor of shape 2*3 filled with 0 (float by default)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# ATTRIBUTES OF A TENSOR
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}") # Tells where the tensors data is stored


# OPERATIONS ON TENSORS
# We move our tensor to the current accelerator if available
if torch.accelerator.is_available(): # Checks if GPU is available
    tensor = tensor.to(torch.accelerator.current_accelerator()) # Moves tensor to GPU

# # If you want a device-agnostic approach (works for CPU or GPU without changing code):
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor = tensor.to(device)


tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
#print(f"Last column: {tensor[:, -1]}")
tensor[:,1] = 0 # Replacing second columns with 0's
print(tensor)

# JOINING TENSORS
t1 = torch.cat([tensor, tensor, tensor], dim=1) # dim=0 :- Vertical Stacking   dim=1 :- Horizontally Stacking
print(t1)



#ARITHMETIC OPERATIONS

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T  # Matrix Multiplication
y2 = tensor.matmul(tensor.T) #Matrix Multiplication in Pytorch

y3 = torch.rand_like(y1) # Creates a tensor of same shape and dtype as y1 but filled with random numbers
torch.matmul(tensor, tensor.T, out=y3) # This is same as @ and tensor.matmul (matrix multiplication)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor) # Element wise multiplication

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3) # Element wise multiplication


# Single Element Tensor
agg = tensor.sum() # Summing up all the elements of a tensor
agg_item = agg.item() # Extracts the python number from the tensor
print(agg_item, type(agg_item))

# In place operations
print(f"{tensor} \n")
tensor.add_(5) # Adds value 5 to all the elements of the tensor
print(tensor)


# BRIDGE WITH NUMPY


# TENSOR TO NUMPY ARRAY
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.
t[0] = 42 # change the first element of the tensor to 42, it will be reflected in numpy array also
print(f"t after change: {t}")
print(f"n after change: {n}")

t.add_(1) # add 1 to all elements of the tensor
print(f"t: {t}")
print(f"n: {n}")

# NUMPY ARRAY TO TENSOR
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n) # This is a numpy operation where 1 is added to each element in n
print(f"t: {t}")
print(f"n: {n}")