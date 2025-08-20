import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True) # Creates a random weight matrix of shape (5, 3)
print(w)
b = torch.randn(3, requires_grad=True) # Creates a random bias vector of shape (3,1)
print(b)
z = torch.matmul(x, w)+b # Linear Multiplication along with addition of bias
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # Calculating the cross entropy loss

print(f"Gradient function for z = {z.grad_fn}") # .grad_fn tells you which operation created the tensor.
print(f"Gradient function for loss = {loss.grad_fn}") # Cross entropy loss

loss.backward() # BackPropagation
print(w.grad) # Prints the gradient tensor for w, with shape (5, 3).
print(b.grad) # Prints the gradient tensor for b, with shape (3,).


# Disabling Gradient Tracking
z = torch.matmul(x, w)+b # w and b were created with requires_grad=True. x does not require gradients (requires_grad=False).
# When you do torch.matmul(x, w) + b, the result z is built from tensors that require gradients. By default, PyTorch sets requires_grad=True for the result (because it depends on w and b).
print(z.requires_grad) # Therefore it prints True

with torch.no_grad(): # Pytorch cannot attach any gradients to this ie. it wont be tracked back
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Different Method to disable
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)



# Tensor Gradients and Jacobian (Optional)
inp = torch.eye(4, 5, requires_grad=True)  # Creates a 4×5 identity matrix (diagonal = 1, rest = 0).
out = (inp+1).pow(2).t() # (inp + 1) → adds 1 to every element. .pow(2) → squares each element. .t() → transposes the result (swaps rows and columns).
# (out.sum()).backward() #Same
out.backward(torch.ones_like(out), retain_graph=True) # torch.ones_like(out) makes a tensor of ones with the same shape as out (5×4).
# out is a matrix of values, not just one number You can’t directly call .backward() on a non-scalar
# So what we do is we take sum of all the elements present in the matrix
print(f"First call\n{inp.grad}") # Prints the gradients after the first backward pass.
out.backward(torch.ones_like(out), retain_graph=True)  # Runs backprop a second time without clearing gradients
print(f"\nSecond call\n{inp.grad}") # By default, PyTorch accumulates gradients into inp.grad.
inp.grad.zero_() #.zero_() sets inp.grad to zero in-place.
out.backward(torch.ones_like(out), retain_graph=True) # Back-prop again
print(f"\nCall after zeroing gradients\n{inp.grad}") # This will contain only fresh gradients from the backward pass