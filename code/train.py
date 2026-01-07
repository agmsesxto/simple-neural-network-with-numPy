import numpy as np

np.set_printoptions(
    threshold=np.inf,
    suppress=True,
)

i = int(0)

# ---- Funciones de activación ----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def calculate(x):
    z1 = x @ W1
    a1 = sigmoid(z1)
    z2 = a1 @ W2
    a2 = sigmoid(z2)
    z3 = a2 @ W3
    a3 = sigmoid(z3)
    return(a3)

# ---- Datos de entrenamiiento ----
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# ---- Definir red ----
np.random.seed(42)

W1 = np.random.randn(2, 10)
W2 = np.random.randn(10, 10)
W3 = np.random.randn(10, 1)

# ---- Entrenamiento ----
lr = 0.1
epochs = 100000

while i <= epochs:
    # ---- Forward pass ----
    z1 = X @ W1
    a1 = sigmoid(z1)
    
    z2 = a1 @ W2
    a2 = sigmoid(z2)
    
    z3 = a2 @ W3
    a3 = sigmoid(z3) # salida final
    
    # ---- Backpropagation ----
    error = y - a3
    d3 = error * sigmoid_deriv(a3)
    
    error2 = d3 @ W3.T
    d2 = error2 * sigmoid_deriv(a2)
    
    error1 = d2 @ W2.T
    d1 = error1 * sigmoid_deriv(a1)
    
    # ---- Actualización de peso----
    W3 += a2.T @ d3 * lr
    W2 += a1.T @ d2 * lr
    W1 += X.T @ d1 * lr
    
    porcentaje = (i * 100) / epochs
    if porcentaje.is_integer():
        print(int(porcentaje))
    i = i + 1
i = 1

print(calculate(X))
input()

# ---- Guardar pesos de la red ----
with open("weights/pesos_W1.txt", "w") as f:
    f.write(str(W1))

with open("weights/pesos_W2.txt", "w") as f:
    f.write(str(W2))

with open("weights/pesos_W3.txt", "w") as f:
    f.write(str(W3))

np.save("weights/W1.npy", W1)
np.save("weights/W2.npy", W2)
np.save("weights/W3.npy", W3)