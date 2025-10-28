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
]),
"R": np.array([
 [ 1,  1,  1,  1, -1],
 [ 1, -1, -1, -1,  1],
 [ 1,  1,  1,  1, -1],
 [ 1, -1,  1, -1, -1],
 [ 1, -1, -1,  1, -1]
]),
"P": np.array([
 [ 1,  1,  1,  1, -1],
 [ 1, -1, -1, -1,  1],
 [ 1,  1,  1,  1, -1],
 [ 1, -1, -1, -1, -1],
 [ 1, -1, -1, -1, -1]
]),
"F": np.array([
 [ 1,  1,  1,  1,  1],
 [ 1, -1, -1, -1, -1],
 [ 1,  1,  1,  1, -1],
 [ 1, -1, -1, -1, -1],
 [ 1, -1, -1, -1, -1]
]),
"E": np.array([
 [ 1,  1,  1,  1,  1],
 [ 1, -1, -1, -1, -1],
 [ 1,  1,  1,  1, -1],
 [ 1, -1, -1, -1, -1],
 [ 1,  1,  1,  1,  1]
]),
"B": np.array([
 [ 1,  1,  1,  1, -1],
 [ 1, -1, -1, -1,  1],
 [ 1,  1,  1,  1, -1],
 [ 1, -1, -1, -1,  1],
 [ 1,  1,  1,  1, -1]
]),
"H": np.array([
 [ 1, -1, -1, -1,  1],
 [ 1, -1, -1, -1,  1],
 [ 1,  1,  1,  1,  1],
 [ 1, -1, -1, -1,  1],
 [ 1, -1, -1, -1,  1]
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

    def energy(self, states):
        s = np.array(states)
        return -0.5 * np.dot(s.T, np.dot(self.weights, s))

    def predict(self, x, epochs):
        self.states = x.copy()
        prev_states = x.copy()
        energies = []
        # For the fixed number of epochs:
        for epoch in range(epochs):
            z = np.dot(self.weights, self.states)
            self.states = [self.activation_function(zi) for zi in z]
            print("Epoch", epoch)
            print(np.array(self.states) - np.array(prev_states))

            # Calcular energía actual
            E = self.energy(self.states)
            energies.append(E)

            # Mostrar patrón actual
            arr_states = np.array(self.states)
            mostrar_patron(
                arr_states.reshape(int(np.sqrt(self.n_inputs)), -1),
                titulo=f"Época {epoch}"
            )

            # Comparación elemento a elemento
            if np.array_equal(self.states, prev_states):
                print("Convergencia alcanzada en epoch", epoch)
                return self.states, energies
            prev_states = self.states.copy()

        return self.states, energies


A = LETTER_PATTERNS["A"].flatten()
B = LETTER_PATTERNS["B"].flatten()
C = LETTER_PATTERNS["C"].flatten()
D = LETTER_PATTERNS["D"].flatten()
E = LETTER_PATTERNS["E"].flatten()
F = LETTER_PATTERNS["F"].flatten()
H = LETTER_PATTERNS["H"].flatten()
P = LETTER_PATTERNS["P"].flatten()
R = LETTER_PATTERNS["R"].flatten()


"""
X = np.array([A, B, C, D])

perceptron = Perceptron(n_inputs=25)
perceptron.train(X)



# Tomemos la letra A del diccionario anterior
A = LETTER_PATTERNS["F"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.6
A_ruidosa = agregar_ruido(A, nivel_ruido=nivel_ruido)

mostrar_patron(A, "Letra A original")
mostrar_patron(A_ruidosa, f"Letra A con {nivel_ruido*100}% de ruido")


y, energies = perceptron.predict(A_ruidosa.flatten(), epochs=10)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")

# Graficar energía en función de la época
plt.figure()
plt.plot(range(len(energies)), energies, marker='o')
plt.xlabel("Época")
plt.ylabel("Energía de Hopfield")
plt.title("Energía vs Época")
plt.grid(True)
plt.show()
"""

"""
#ESPUREO


# Tomemos la letra A del diccionario anterior
A = LETTER_PATTERNS["A"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.3
A_ruidosa = agregar_ruido(A, nivel_ruido=nivel_ruido)

mostrar_patron(A, "Letra A original")
mostrar_patron(A_ruidosa, f"Letra A con {nivel_ruido*100}% de ruido")


y = perceptron.predict(A_ruidosa.flatten(), epochs=10)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")
"""

"""
# Otras letras

X = np.array([A, R, P, F, E, B, H])

perceptron = Perceptron(n_inputs=25)
perceptron.train(X)

# Tomemos la letra R del diccionario anterior
R = LETTER_PATTERNS["A"]


# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.2
R_ruidosa = agregar_ruido(R, nivel_ruido=nivel_ruido)

mostrar_patron(R, "Letra R original")
mostrar_patron(R_ruidosa, f"Letra R con {nivel_ruido*100}% de ruido")


y = perceptron.predict(R_ruidosa.flatten(), epochs=10)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")
"""

"""
# Letra F en red entrenada con A, B, C y D, oscilando entre 2 estados

X = np.array([A, B, C, D])

perceptron = Perceptron(n_inputs=25)
perceptron.train(X)

# Tomemos la letra F del diccionario anterior
F = LETTER_PATTERNS["F"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.6
F_ruidosa = agregar_ruido(F, nivel_ruido=nivel_ruido)

mostrar_patron(F, "Letra F original")
mostrar_patron(F_ruidosa, f"Letra F con {nivel_ruido*100}% de ruido")


y, energies = perceptron.predict(F_ruidosa.flatten(), epochs=10)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")

# Graficar energía en función de la época
plt.figure()
plt.plot(range(len(energies)), energies, marker='o')
plt.xlabel("Época")
plt.ylabel("Energía de Hopfield")
plt.title("Energía vs Época")
plt.grid(True)
plt.show()
"""

# Letra F en red entrenada con A, B, C y D, que converge a C

X = np.array([A, B, C, D])

perceptron = Perceptron(n_inputs=25)
perceptron.train(X)

# Tomemos la letra F del diccionario anterior
F = LETTER_PATTERNS["F"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.5
F_ruidosa = agregar_ruido(F, nivel_ruido=nivel_ruido)

mostrar_patron(F, "Letra F original")
mostrar_patron(F_ruidosa, f"Letra F con {nivel_ruido*100}% de ruido")


y, energies = perceptron.predict(F_ruidosa.flatten(), epochs=10)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")

# Graficar energía en función de la época
plt.figure()
plt.plot(range(len(energies)), energies, marker='o')
plt.xlabel("Época")
plt.ylabel("Energía de Hopfield")
plt.title("Energía vs Época")
plt.grid(True)
plt.show()

# Probar ir agrandando la red