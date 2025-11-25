# %%
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
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

# mostrar_emoji(X, 0)
# mostrar_emoji(X, 1)
# mostrar_emoji(X, 2)
# mostrar_emoji(X, 3)
# mostrar_emoji(X, 4)
# mostrar_emoji(X, 5)
# mostrar_emoji(X, 6)
# mostrar_emoji(X, 7)
# mostrar_emoji(X, 8)
#mostrar_emoji(X, 9)

emoji_labels = np.array([f"E{i}" for i in range(len(X))])

X = X.reshape((X.shape[0], -1))

# %%
