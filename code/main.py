import numpy as np

np.set_printoptions(
    threshold=np.inf,
    suppress=True,
)

# ---- Definir red y cargar pesos ----
model = np.load("model.npy", allow_pickle=True).item()
W1 = model["W1"]
W2 = model["W2"]
W3 = model["W3"]


# ---- Funciones de acivación ----
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

while 1 == 1:
	X[0, 0] = int(input("Primer numero: "))
	X[0, 1] = int(input("Segundo numero: "))
	print(predict(X))