import numpy as np
import matplotlib.pyplot as plt

e = 2.71
z = np.linspace(-10,10,100)
#print(z)

#Sigmoid Function
sig = 1/(1+e**(-z))
# print(sig)

#TanH Function
tan = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
# print(tan)
# print(type(tan))

#TanH Function Derivative
tan_der = 1 - tan**2
# print(tan_der)

#Plotting both the functions
plt.plot(z, tan, label='tanh(z)', color='blue', linewidth=2)
plt.plot(z, tan_der, label="tanh'(z)", color='red', linestyle='--', linewidth=2)
plt.title("Tanh Function and its Derivative")
plt.xlabel("z")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
# plt.show()

#ReLu Function
relu = [i if i > 0 else 0 for i in z ]
relu = np.array(relu)
# print(relu)

#Derivative for ReLu Activation function
relu_der = [0 if i < 0 else 1 for i in z]
# print(relu_der)

#Plotting both the functions
plt.plot(z, relu, label='ReLu(z)', color='blue', linewidth=2)
plt.plot(z, relu_der, label="ReLu'(z)", color='red', linestyle='--', linewidth=2)
plt.title("ReLu Function and its Derivative")
plt.xlabel("z")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
# plt.show()

#Leaky ReLu Function
alpha = 0.1 #Usual value is 0.01, kept 0.1 for visualization only
L_relu = [i if i > 0 else alpha * i for i in z ]
L_relu = np.array(L_relu)
# print(L_relu)

# Derivative for leaky ReLu
L_relu_der = [alpha if i < 0 else 1 for i in z]
# print(L_relu_der)

#Plotting both the functions
plt.plot(z, L_relu, label='L_ReLu(z)', color='blue', linewidth=2)
plt.plot(z, L_relu_der, label="L_ReLu'(z)", color='red', linestyle='--', linewidth=2)
plt.title("Leaky ReLu Function and its Derivative")
plt.xlabel("z")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
# plt.show()

#Softmax Activation
soft_vec = []
den = sum([np.exp(j) for j in z])
for i in z:
    soft = np.exp(i)/(den)
    soft_vec.append(soft)

soft_vec = np.array(soft_vec)
# print(soft_vec)
# print(sum(soft_vec))

# Derivative of Softmax Activation
# z = [2.0,1.0,0.1]
den = sum([np.exp(j) for j in z])
Jacob = []
for i in range (len(z)):
    J_row = []
    for j in range (len(z)):
        if i == j:
            Si = np.exp(z[i])/den
            d_Si = Si*(1-Si)
            J_row.append(d_Si)
        else:
            Si = np.exp(z[i])/den
            Sj = np.exp(z[j])/den
            d_Sij = -Si*Sj
            J_row.append(d_Sij)
    Jacob.append(J_row)
# print(Jacob)

#Plotting both the functions
plt.plot(z, soft_vec, label='Softmax(z)', color='blue', linewidth=2)
# plt.plot(z, L_relu_der, label="L_ReLu'(z)", color='red', linestyle='--', linewidth=2)
plt.title("Softmax Activation Function and its Derivative")
plt.xlabel("z")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
# plt.show()

print(f"The minimum values of the function are: \n"
      f"Tan-H : {min(tan)} \n"
      f"ReLu : {min(relu)} \n"
      f"Leaky ReLu : {min(L_relu)} \n"
      f"Softmax : {min(soft_vec)} \n")

print(f"The Maximum values of the function are: \n"
      f"Tan-H : {max(tan)} \n"
      f"ReLu : {max(relu)} \n"
      f"Leaky ReLu : {max(L_relu)} \n"
      f"Softmax : {max(soft_vec)} \n")

print(f"Is the output of the function zero centered: \n"  #Earlier we used to do standardization and same logic holds good here
      f"Tan-H : {np.mean(tan)} : Yes \n"
      f"ReLu : {np.mean(relu)} :  No \n"
      f"Leaky ReLu : {np.mean(L_relu)} : No \n"
      f"Softmax : {np.mean(soft_vec)} :  Almost Yes \n")

print(f"What happens to the gradient when the input values are too small or too big: \n"
      f"Tan-H : The values doesn't change much \n"
      f"ReLu : {max(relu)} \n"
      f"Leaky ReLu : {max(L_relu)} \n"
      f"Softmax : {max(soft_vec)} \n")









