import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

            # Calcular energía actual
            E = self.energy(self.states)
            energies.append(E)

            z = np.dot(self.weights, self.states)
            self.states = [self.activation_function(zi) for zi in z]
            print("Epoch", epoch+1)
            print(np.array(self.states) - np.array(prev_states))



            # Mostrar patrón actual
            # arr_states = np.array(self.states)
            # mostrar_patron(
            #     arr_states.reshape(int(np.sqrt(self.n_inputs)), -1),
            #     titulo=f"Época {epoch+1}"
            # )

            # Comparación elemento a elemento
            if np.array_equal(self.states, prev_states):
                print("Convergencia alcanzada en epoch", epoch+1)
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



X = np.array([A, B, C, D])

perceptron = Perceptron(n_inputs=25)
perceptron.train(X)



# Tomemos la letra A del diccionario anterior
A = LETTER_PATTERNS["A"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.05
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
#plt.figure(figsize=(12,8))
plt.plot(range(len(energies)), energies, marker='o')
plt.xlabel("Época")
plt.ylabel("Energía de Hopfield")
plt.title("Energía vs Época")
plt.grid(True)
plt.show()



"""
#ESPUREO


# Tomemos la letra A del diccionario anterior
A = LETTER_PATTERNS["A"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.3
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
#plt.figure(figsize=(12,8))
plt.plot(range(len(energies)), energies, marker='o')
plt.xlabel("Época")
plt.ylabel("Energía de Hopfield")
plt.title("Energía vs Época")
plt.grid(True)
plt.show()
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


y, energies = perceptron.predict(R_ruidosa.flatten(), epochs=10)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")
"""

"""
# Letra H en red entrenada con A, B, C y D, oscilando entre 2 estados

X = np.array([A, B, C, D])

perceptron = Perceptron(n_inputs=25)
perceptron.train(X)

# Tomemos la letra F del diccionario anterior
H = LETTER_PATTERNS["H"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0.5
H_ruidosa = agregar_ruido(H, nivel_ruido=nivel_ruido)

mostrar_patron(H, "Letra H original")
mostrar_patron(H_ruidosa, f"Letra H con {nivel_ruido*100}% de ruido")


y, energies = perceptron.predict(H_ruidosa.flatten(), epochs=10)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")

# Graficar energía en función de la época
plt.figure(figsize=(12,8))
plt.plot(range(len(energies)), energies, marker='o')
plt.xlabel("Época")
plt.ylabel("Energía de Hopfield")
plt.title("Energía vs Época")
plt.grid(True)
plt.show()
"""

"""
# Letra H en red entrenada con A, B, C y D, que converge a A

X = np.array([A, B, C, D])

perceptron = Perceptron(n_inputs=25)
perceptron.train(X)

# Tomemos la letra F del diccionario anterior
H = LETTER_PATTERNS["H"]

# Generamos una versión ruidosa (% de los bits invertidos)
nivel_ruido = 0
H_ruidosa = agregar_ruido(H, nivel_ruido=nivel_ruido)

mostrar_patron(H, "Letra H original")
mostrar_patron(H_ruidosa, f"Letra H con {nivel_ruido*100}% de ruido")


y, energies = perceptron.predict(H_ruidosa.flatten(), epochs=10)
print(y)

# Convertir la salida (lista o vector) en una matriz 5x5
y_matrix = np.array(y).reshape(5, 5)

# Mostrar la letra reconstruida
mostrar_patron(y_matrix, "Letra reconstruida por el perceptrón")

# Graficar energía en función de la época
plt.figure(figsize=(12,8))
plt.plot(range(len(energies)), energies, marker='o')
plt.xlabel("Época")
plt.ylabel("Energía de Hopfield")
plt.title("Energía vs Época")
plt.grid(True)
plt.show()
"""

############## Análisis almacenamiento ##############

### Construcción de patrones ortogonales 10x10
def _legendre_symbol(a, p):
    a %= p
    if a == 0:
        return 0
    ls = pow(a, (p-1)//2, p)
    return 1 if ls == 1 else -1

def _paley_hadamard(q=19):
    # Paley tipo I: q primo con q ≡ 3 (mod 4). 19 funciona => matriz 20×20
    p = q
    n = q + 1
    A = np.zeros((q, q), dtype=int)
    for i in range(q):
        for j in range(q):
            if i == j:
                A[i, j] = 0
            else:
                A[i, j] = _legendre_symbol(i - j, p)
    J = np.ones((q, 1), dtype=int)
    H = np.vstack([np.ones((1, n), dtype=int), np.concatenate([J, A], axis=1)])
    # Convertimos los ceros de la diagonal de A en +1 para tener entradas ±1
    H[1:, 1:][np.eye(q, dtype=bool)] = 1
    return H  # 20×20, filas (excepto la 0) son mutuamente ortogonales

def build_orthogonal_10x10_patterns(num=5, reshape=(10, 10)):
    H = _paley_hadamard(19)   # 20×20
    rows = H[1:1+num, :]      # tomamos 5 filas (todas ortogonales entre sí)
    w = np.ones(5, dtype=int) # vector de 5 unos para expandir a 100 (=20×5)
    vecs = [np.kron(r, w) for r in rows]  # cada uno queda de longitud 100 (±1)
    mats = [v.reshape(reshape) for v in vecs]
    return mats
###

# # === Ejemplo de uso ===
patterns_10x10 = build_orthogonal_10x10_patterns(num=5)

# # Verificación: productos punto deben ser 0
# flat = [p.ravel() for p in patterns_10x10]
# for i in range(len(flat)):
#     for j in range(i+1, len(flat)):
#         print(f"P{i+1}·P{j+1} =", int(flat[i] @ flat[j]))
# print("Producto punto entre P1 y P2 (debe ser 0):")
# print(np.dot(np.array(patterns_10x10[0]).flatten(), np.array(patterns_10x10[1]).flatten())) 

# # # === Visualizarlos ===
# fig, axes = plt.subplots(1, 5, figsize=(15, 3))
# for i, ax in enumerate(axes):
#     ax.imshow(patterns_10x10[i], cmap='gray_r', vmin=-1, vmax=1)
#     ax.set_title(f"Pattern {i+1}")
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

### Construcción de patrones ortogonales 4x4
def hadamard4():
    """Hadamard(4) via Sylvester."""
    H2 = np.array([[1, 1],
                   [1, -1]], dtype=int)
    H4 = np.kron(H2, H2)  # 4x4
    return H4

def build_orthogonal_4x4_patterns(num=5):
    """
    Devuelve 'num' patrones 4x4 ±1 mutuamente ortogonales al aplanar.
    Se generan como outer(u_i, v_j) con u_i, v_j filas de Hadamard(4).
    Hay 4×4 = 16 patrones ortogonales disponibles.
    """
    H = hadamard4()                 # 4x4, filas ortogonales
    pairs = [(i, j) for i in range(4) for j in range(4)]  # 16 combinaciones
    if num > 16:
        raise ValueError("Máximo num=16 para 4x4.")
    patterns = []
    for k in range(num):
        i, j = pairs[k]
        u = H[i, :]                 # (4,)
        v = H[j, :]                 # (4,)
        P = np.outer(u, v)          # (4,4), valores ±1
        patterns.append(P.astype(int))
    return patterns
###

# === Ejemplo de uso ===
patterns_4x4 = build_orthogonal_4x4_patterns(num=16)

# # Verificación de ortogonalidad (productos punto entre aplanados)
# flat = [p.ravel() for p in patterns_4x4]
# for i in range(len(flat)):
#     for j in range(i+1, len(flat)):
#         print(f"P{i+1}·P{j+1} =", int(flat[i] @ flat[j]))  # => todos 0

# === Visualización monocromática (-1 blanco, +1 negro) ===
mono_cmap = LinearSegmentedColormap.from_list("mono_bw", ["white", "black"])

n = len(patterns_4x4)
rows = 4
cols = int(np.ceil(n / rows))

fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows), dpi=150)
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < n:
        ax.imshow(patterns_4x4[i], cmap=mono_cmap, vmin=-1, vmax=1)
        ax.set_title(f"P{i+1}", fontsize=10)
    ax.axis("off")

plt.subplots_adjust(hspace=0.55)  # 🔹 separaciones horizontal y vertical
plt.show()


# --------------------------------------------------------# Evaluación de performance con patrones ortogonales 4x4

# === Generar los 16 patrones ortogonales 4x4 ===
patterns_4x4 = build_orthogonal_4x4_patterns(num=16)

def evaluar_performance(num_patterns, epochs=10, verbose=False):
    #Entrena con los primeros 'num_patterns' y evalúa la cantidad reconocida.
    X = np.array([p.flatten() for p in patterns_4x4[:num_patterns]])
    perceptron = Perceptron(n_inputs=16)
    perceptron.train(X)
    correct = 0
    for i, pattern in enumerate(X):
        final_state, _ = perceptron.predict(pattern.copy(), epochs=epochs)
        if np.array_equal(final_state, pattern):
            correct += 1
        elif verbose:
            print(f"Patrón {i+1} no reconocido")
    return correct

# === Evaluar de 1 a 16 patrones ===
ratios = []
num_pat = range(1, 17)

for n in num_pat:
    correct = evaluar_performance(n, epochs=10)
    ratio = correct / n
    ratios.append(ratio)
    print(f"{n} patrones → reconocidos {correct}/{n} = {ratio:.2f}")

# === Graficar la curva de performance ===
plt.figure(figsize=(7,4), dpi=150)
plt.scatter(num_pat, ratios, marker="o", label="Datos simulados")
plt.vlines(0.138*max(num_pat), ymin=0, ymax=1, color='r', linestyle='--', label=r'Límite teórico $\alpha_c$')
plt.xlabel("Número de patrones de entrenamiento")
plt.ylabel("Fracción de patrones reconocidos")
plt.title("Capacidad de almacenamiento de la red de Hopfield (4x4)")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------------# Evaluación de performance con patrones ortogonales 10x10

# === Generar los 16 patrones ortogonales 10x10 ===
patterns_10x10 = build_orthogonal_10x10_patterns(num=100)

def evaluar_performance(num_patterns, epochs=10, verbose=False):
    #Entrena con los primeros 'num_patterns' y evalúa la cantidad reconocida.
    X = np.array([p.flatten() for p in patterns_10x10[:num_patterns]])
    perceptron = Perceptron(n_inputs=100)
    perceptron.train(X)
    correct = 0
    for i, pattern in enumerate(X):
        final_state, _ = perceptron.predict(pattern.copy(), epochs=epochs)
        if np.array_equal(final_state, pattern):
            correct += 1
        elif verbose:
            print(f"Patrón {i+1} no reconocido")
    return correct

# === Evaluar de 1 a 100 patrones ===
ratios = []
num_pat = range(1, 101)

for n in num_pat:
    correct = evaluar_performance(n, epochs=10)
    ratio = correct / n
    ratios.append(ratio)
    print(f"{n} patrones → reconocidos {correct}/{n} = {ratio:.2f}")

# === Graficar la curva de performance ===
plt.figure(figsize=(7,4), dpi=150)
plt.scatter(num_pat, ratios, marker="o", label="Datos simulados")
plt.vlines(0.138*max(num_pat), ymin=0, ymax=1, color='r', linestyle='--', label=r'Límite teórico $\alpha_c$')
plt.xlabel("Número de patrones de entrenamiento")
plt.ylabel("Fracción de patrones reconocidos")
plt.title("Capacidad de almacenamiento de la red de Hopfield (10x10)")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------------# Patrones aleatorios 4x4

def generar_patrones_aleatorios(num_patterns, size=16):
    """
    Genera 'num_patterns' patrones aleatorios de longitud 'size'
    con valores ±1.
    """
    return np.random.choice([-1, 1], size=(num_patterns, size))


def mostrar_patrones_aleatorios(num_patterns=8, lado=4, dpi=150, wspace=0.3, hspace=0.4):
    """
    Genera y muestra 'num_patterns' patrones aleatorios (4x4 por defecto)
    como imágenes monocromáticas, distribuidos en 4 filas.
    """
    patrones = generar_patrones_aleatorios(num_patterns, size=lado**2)

    rows = 4
    cols = int(np.ceil(num_patterns / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows), dpi=dpi)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_patterns:
            ax.imshow(patrones[i].reshape(lado, lado), cmap="gray_r", vmin=-1, vmax=1)
            ax.set_title(f"P{i+1}", fontsize=10)
        ax.axis("off")

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()

    return patrones

#patrones_aleatorios = mostrar_patrones_aleatorios(num_patterns=16, lado=4)

def evaluar_performance_aleatoria(num_patterns, epochs=10, n_imputs=16, size=16):
    """
    Entrena una red Hopfield con 'num_patterns' patrones aleatorios
    y devuelve la fracción de patrones reconocidos correctamente.
    """
    # Generar los patrones
    X = generar_patrones_aleatorios(num_patterns, size=size)
    
    # Entrenar la red
    perceptron = Perceptron(n_inputs=n_imputs)
    perceptron.train(X)
    
    # Evaluar cuántos patrones recupera correctamente
    correct = 0
    for i, pattern in enumerate(X):
        final_state, _ = perceptron.predict(pattern.copy(), epochs=10)
        if np.array_equal(final_state, pattern):
            correct += 1
    return correct / num_patterns


# === Evaluar de 1 a 16 patrones ===
ratios = []
desvios = []
num_pat = range(1, 17)

# Para reducir el ruido estadístico, promediamos varios ensayos por cada N
num_repeticiones = 20

for n in num_pat:
    perfo = [evaluar_performance_aleatoria(n) for _ in range(num_repeticiones)]
    promedio = np.mean(perfo)
    desvio = np.std(perfo)/np.sqrt(num_repeticiones)
    ratios.append(promedio)
    desvios.append(desvio)
    print(f"{n} patrones → fracción promedio reconocida = {promedio:.2f}")

# === Graficar la curva ===
plt.figure(figsize=(7,4), dpi=150)
plt.errorbar(num_pat, ratios, yerr=desvios, fmt='o', capsize=5, label="Datos simulados")
#plt.plot(num_pat, ratios, marker="o")
plt.vlines(0.138*max(num_pat), ymin=0, ymax=1, color='r', linestyle='--', label=r'Límite teórico $\alpha_c$')
plt.xlabel("Número de patrones de entrenamiento")
plt.ylabel("Fracción promedio reconocida")
plt.title("Curva de capacidad de la red de Hopfield (patrones aleatorios 4×4)")
plt.legend()
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()

"""
# === Evaluar de 1 a 100 patrones ===
ratios = []
desvios = []
num_pat = range(1, 101)

# Para reducir el ruido estadístico, promediamos varios ensayos por cada N
num_repeticiones = 20

for n in num_pat:
    perfo = [evaluar_performance_aleatoria(n, n_imputs=100, size=100) for _ in range(num_repeticiones)]
    promedio = np.mean(perfo)
    desvio = np.std(perfo)/np.sqrt(num_repeticiones)
    ratios.append(promedio)
    desvios.append(desvio)
    print(f"{n} patrones → fracción promedio reconocida = {promedio:.2f}")

# === Graficar la curva ===
plt.figure(figsize=(7,4), dpi=150)
plt.errorbar(num_pat, ratios, yerr=desvios, fmt='o', capsize=5, label="Datos simulados")
#plt.plot(num_pat, ratios, marker="o")
plt.vlines(0.138*max(num_pat), ymin=0, ymax=1, color='r', linestyle='--', label=r'Límite teórico $\alpha_c$')
plt.xlabel("Número de patrones de entrenamiento")
plt.ylabel("Fracción promedio reconocida")
plt.title("Curva de capacidad de la red de Hopfield (patrones aleatorios 10×10)")
plt.legend()
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()
"""