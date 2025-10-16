import numpy as np
import matplotlib.pyplot as plt
import random


# fijar semilla para reproducibilidad
SEED = 10
np.random.seed(SEED)
random.seed(SEED)



LETTER_PATTERNS = {
"A": np.array([
 [ -1,  1,  1,  1, -1],
 [  1, -1, -1, -1,  1],
 [  1,  1,  1,  1,  1],
 [  1, -1, -1, -1,  1],
 [  1, -1, -1, -1,  1]
]),
"B": np.array([
 [  1,  1,  1,  1, -1],
 [  1, -1, -1, -1,  1],
 [  1,  1,  1,  1, -1],
 [  1, -1, -1, -1,  1],
 [  1,  1,  1,  1, -1]
]),
"C": np.array([
 [ -1,  1,  1,  1,  1],
 [  1, -1, -1, -1, -1],
 [  1, -1, -1, -1, -1],
 [  1, -1, -1, -1, -1],
 [ -1,  1,  1,  1,  1]
]),
"D": np.array([
 [  1,  1,  1, -1, -1],
 [  1, -1, -1,  1, -1],
 [  1, -1, -1, -1,  1],
 [  1, -1, -1,  1, -1],
 [  1,  1,  1, -1, -1]
])}


def mostrar_patron(p, titulo=""):
    plt.imshow(-p, cmap="gray")
    plt.title(titulo)
    plt.axis("off")
    plt.show()


def agregar_ruido(patron, nivel_ruido=0.2):
    """
    Agrega ruido a un patrón de Hopfield (matriz o vector de 1 y -1).
    """
    patron_flat = patron.flatten()
    n = len(patron_flat)
    n_flip = int(n * nivel_ruido)
    
    # indices aleatorios a invertir
    indices = np.random.choice(n, n_flip, replace=False)
    
    patron_ruidoso = patron_flat.copy()
    patron_ruidoso[indices] *= -1
    
    return patron_ruidoso.reshape(patron.shape)


# Perceptron Class
class Perceptron:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        # Initialize weights matrix
        self.weights = np.zeros((n_inputs, n_inputs))
        # Initialize states vector
        self.states = np.zeros(n_inputs)
    
    # Activation Function
    def activation_function(self, z):
        return 1 if z >= 0 else -1
    
    # Train the perceptron
    def train(self, X):
        # Calculate weights
        for i in range(self.n_inputs):
            for j in range(self.n_inputs):
                if i != j:
                    xi = X[:, i]
                    xj = X[:, j]
                    wij = (1 / self.n_inputs) * np.dot(xi, xj)
                    self.weights[i, j] = wij

    def predict(self, x, epochs):
        self.states = x
        # For the fixed number of epochs:
        for epoch in range(epochs):
            z = np.dot(self.weights, x)
            self.states = [self.activation_function(zi) for zi in z]

        return self.states
    



A = LETTER_PATTERNS["A"].flatten()
B = LETTER_PATTERNS["B"].flatten()
C = LETTER_PATTERNS["C"].flatten()
D = LETTER_PATTERNS["D"].flatten()

X = np.array([A, B, C, D])

perceptron = Perceptron(n_inputs=25)
perceptron.train(X)



# Tomemos la letra A del diccionario anterior
A = LETTER_PATTERNS["A"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.2
A_ruidosa = agregar_ruido(A, nivel_ruido=nivel_ruido)

mostrar_patron(A, "Letra A original")
mostrar_patron(A_ruidosa, f"Letra A con {nivel_ruido*100}% de ruido")


y = perceptron.predict(A_ruidosa.flatten(), epochs=10000)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")


"""
ESPUREO


# Tomemos la letra A del diccionario anterior
A = LETTER_PATTERNS["A"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.3
A_ruidosa = agregar_ruido(A, nivel_ruido=nivel_ruido)

mostrar_patron(A, "Letra A original")
mostrar_patron(A_ruidosa, f"Letra A con {nivel_ruido*100}% de ruido")


y = perceptron.predict(A_ruidosa.flatten(), epochs=1000)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")

"""