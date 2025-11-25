# %%
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from constants2 import *
np.set_printoptions(threshold=np.inf)

def crear_dataset_emojis(carpeta, n_emojis, resolucion=(32, 32), threshold=128, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    archivos = sorted(
        [f for f in os.listdir(carpeta) if f.lower().endswith(".png") and f.startswith("E")]
    )

    if len(archivos) == 0:
        raise ValueError("No se encontraron archivos 'E*.png' en la carpeta.")

    elegidos = random.sample(archivos, n_emojis)

    dataset = []

    for archivo in elegidos:
        path = os.path.join(carpeta, archivo)
        # Cargar imagen en blanco y negro
        img = Image.open(path).convert("L")
        img = img.resize(resolucion, Image.LANCZOS)

        arr = np.array(img, dtype=np.uint8)

        binario = (arr >= threshold).astype(np.float32) 
        binario = binario * 2 - 1

        dataset.append(binario)

    return np.stack(dataset)

def mostrar_emoji(dataset, indice):
    if indice < 0 or indice >= len(dataset):
        raise ValueError("Índice fuera de rango.")

    plt.imshow(dataset[indice], cmap="gray")
    plt.axis("off")
    plt.show()

X = crear_dataset_emojis("sia-tp5-master\emojis-x1-32x32", n_emojis=N_EMOJIS, resolucion=(32, 32))

def mostrar_grid_emojis(X, emoji_labels, resolucion=(32,32)):
    N = len(X)

    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)

    plt.figure(figsize=(cols * 1.5, rows * 1.5))

    for i in range(N):
        ax = plt.subplot(rows, cols, i + 1)
        img = X[i].reshape(resolucion)
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(emoji_labels[i], fontsize=8)

    plt.tight_layout()
    plt.show()

emoji_labels = np.array([f"E{i}" for i in range(len(X))])
mostrar_grid_emojis(X, emoji_labels)

X = X.reshape((X.shape[0], -1))

# %%
