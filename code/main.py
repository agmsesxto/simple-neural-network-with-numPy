import numpy as np

W1 = np.load("W1.npy")
W2 = np.load("W2.npy")
W3 = np.load("W3.npy")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x):
    z1 = x @ W1
    a1 = sigmoid(z1)
    z2 = a1 @ W2
    a2 = sigmoid(z2)
    z3 = a2 @ W3
    a3 = sigmoid(z3)
    return(a3)

X = np.array([[0, 0]])

while 1:
	X[0, 0] = int(input("Primer numero: "))
	X[0, 1] = int(input("Segundo numero: "))
	print(predict(X))