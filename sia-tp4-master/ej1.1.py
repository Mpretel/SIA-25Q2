import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.decomposition import PCA


# fijar semilla para reproducibilidad
np.random.seed(10)


# 1. Cargar y preparar datos
dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(dir, "europe.csv"))
countries = df["Country"]
X = df.drop(columns=["Country"])

# Estandarizar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 2. Definir parámetros de la red Kohonen
k = 2
n = X_scaled.shape[1]

n_iterations = 500*n
initial_lr = 0.5 # < 1
initial_r = k / 2 

# Inicializar pesos aleatoriamente
#weights = np.random.rand(k, k, n) # aleatoriamente
# con muestras de entrada para evitar neuronas muertas
weights = np.zeros((k, k, n))
samples = X_scaled[np.random.choice(X_scaled.shape[0], k * k, replace=False)]
for idx, sample in enumerate(samples):
    i, j = divmod(idx, k)
    weights[i, j, :] = sample

# 3. Funciones auxiliares
def find_bmu(x, weights):
    """Encuentra la neurona ganadora (BMU) para un vector x"""
    distances = np.linalg.norm(weights - x, axis=2)
    return np.unravel_index(np.argmin(distances), distances.shape)

def neighborhood_function(bmu_idx, r, k):
    """Devuelve una matriz con el factor de vecindad gaussiano"""
    x, y = np.meshgrid(np.arange(k), np.arange(k))
    dist2 = (x - bmu_idx[0])**2 + (y - bmu_idx[1])**2
    return np.exp(-dist2 / (2 * r**2))


# 4. Entrenamiento de la red
for t in range(n_iterations):
    # Decaimiento de tasa de aprendizaje y radio de vecindad
    lr = initial_lr * np.exp(-t / n_iterations)
    r = initial_r * np.exp(-t / (n_iterations / np.log(initial_r)))

    # Elegir una muestra aleatoria
    x = X_scaled[np.random.randint(0, X_scaled.shape[0])]

    # Neurona ganadora
    bmu_idx = find_bmu(x, weights)

    # Función de vecindad
    h = neighborhood_function(bmu_idx, r, k)

    # Actualizar pesos
    for i in range(k):
        for j in range(k):
            weights[i, j, :] += lr * h[j, i] * (x - weights[i, j, :])



# 5. Asignar cada país a su neurona ganadora
bmu_positions = np.array([find_bmu(x, weights) for x in X_scaled])

# Convertir a índices lineales (por conveniencia)
bmu_linear = np.ravel_multi_index(bmu_positions.T, (k, k))
df_clusters = pd.DataFrame({"Country": countries, "Cluster": bmu_linear})
print(df_clusters.sort_values("Cluster"))


# Reducir a 2D con PCA solo para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=bmu_linear, cmap="tab20", s=100, edgecolor='k')
for i, name in enumerate(countries):
    plt.text(X_pca[i,0]+0.02, X_pca[i,1], name, fontsize=8)
plt.title("Agrupamiento de países según el SOM (en espacio PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Neurona (Cluster)")
plt.show()


# 6. Graficar el mapa U-Matrix (distancias entre neuronas vecinas)
u_matrix = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        neighbors = []
        if i > 0: 
            neighbors.append(weights[i-1, j])
        if i < k-1: 
            neighbors.append(weights[i+1, j])
        if j > 0: 
            neighbors.append(weights[i, j-1])
        if j < k-1: 
            neighbors.append(weights[i, j+1])
        dists = [np.linalg.norm(weights[i, j] - n) for n in neighbors]
        u_matrix[i, j] = np.mean(dists)
plt.figure(figsize=(8, 6))
plt.imshow(u_matrix.T, cmap="coolwarm", origin="lower")
plt.title("Mapa U-Matrix (distancias promedio entre neuronas)")
plt.colorbar(label="Distancia promedio")
# Agrupar países por neurona
neuron_to_countries = {}
for country, (x, y) in zip(countries, bmu_positions):
    neuron_to_countries.setdefault((x, y), []).append(country)
# Dibujar etiquetas distribuidas
for (x, y), names in neuron_to_countries.items():
    n = len(names)
    # Si hay varias etiquetas, las distribuimos en vertical dentro de la celda
    offsets = np.linspace(-0.3, 0.3, n)
    for offset, name in zip(offsets, names):
        plt.text(x, y + offset, name,fontsize=7, ha="center", va="center", color="black")
plt.show()


# 7. Gráfico de ocupación (cantidad de países por neurona)
counts = Counter(map(tuple, bmu_positions))

plt.figure(figsize=(8, 6))
plt.imshow(u_matrix.T, cmap="coolwarm", origin="lower")
for i in range(k):
    for j in range(k):
        plt.text(i, j, counts.get((i, j), 0), color="black", ha="center", va="center")
plt.title("Cantidad de países asociados a cada neurona")
plt.colorbar(label="Distancia promedio")
plt.show()


# 7b. Gráfico de ocupación (cant de paises por neurona) con colores en funcion de la cantidad
u_matrix = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        u_matrix[i, j] = counts.get((i, j), 0)
plt.figure(figsize=(8, 6))
plt.imshow(u_matrix.T, cmap="coolwarm", origin="lower")
for i in range(k):
    for j in range(k):
        plt.text(i, j, counts.get((i, j), 0), color="black", ha="center", va="center")
plt.title("Cantidad de países asociados a cada neurona")
plt.colorbar(label="Cantidad de países")
plt.show()



# 8. Mapas individuales por variable
variables = X.columns
n_vars = len(variables)

# Crear una matriz para acumular los valores promedio por neurona
feature_maps = np.zeros((k, k, n_vars))
for i in range(k):
    for j in range(k):
        indices = np.where((bmu_positions[:, 0] == i) & (bmu_positions[:, 1] == j))[0]
        if len(indices) > 0:
            feature_maps[i, j, :] = X_scaled[indices].mean(axis=0)
        else:
            feature_maps[i, j, :] = np.nan  # neuronas sin muestras asignadas

# Agrupar países por neurona (igual que en el mapa U-Matrix)
neuron_to_countries = {}
for country, (x, y) in zip(countries, bmu_positions):
    neuron_to_countries.setdefault((x, y), []).append(country)

# Graficar todas las variables
fig, axes = plt.subplots(2, int(np.ceil(n_vars / 2)), figsize=(14, 7))
axes = axes.flatten()

for idx, var in enumerate(variables):
    ax = axes[idx]
    im = ax.imshow(feature_maps[:, :, idx].T, cmap="coolwarm", origin="lower")
    ax.set_title(var)
    ax.set_xticks([])
    ax.set_yticks([])

    # Agregar etiquetas de países en cada neurona
    for (x, y), names in neuron_to_countries.items():
        n = len(names)
        offsets = np.linspace(-0.3, 0.3, n)  # desplazar si hay varios países
        for offset, name in zip(offsets, names):
            ax.text(x, y + offset, name, fontsize=6, ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Eliminar subplots vacíos si sobran
for i in range(idx + 1, len(axes)):
    fig.delaxes(axes[i])

plt.suptitle("Distribución de cada variable en el mapa SOM", fontsize=14)
plt.tight_layout()
plt.show()
