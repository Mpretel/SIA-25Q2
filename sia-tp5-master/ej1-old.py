import numpy as np
import re
import os
from autoencoder import Autoencoder, MLP

def font_to_binary_matrix(path="font.h"):
    import re
    import numpy as np

    with open(path, "r") as f:
        content = f.read()

    # Buscar los bloques { ... }
    pattern = re.compile(r"\{([^\}]*)\}")
    matches = pattern.findall(content)

    font_data = []
    for match in matches:
        # Extraer los hexadecimales de cada bloque
        hex_values = re.findall(r"0x[0-9A-Fa-f]+", match)
        if len(hex_values) == 7:  # cada carácter tiene 7 filas
            pattern_bits = []
            for hex_val in hex_values:
                row = int(hex_val, 16)  # 🔹 Convertir de str a entero base 16
                bits = [(row >> i) & 1 for i in range(5)]  # 5 columnas
                pattern_bits.extend(bits[::-1])  # invertir para orden natural
            font_data.append(pattern_bits)

    return np.array(font_data, dtype=float)

import os

# Ruta del archivo font.h en el mismo directorio que el script
script_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(script_dir, "font.h")

def load_font_from_header(path):
    with open(path, "r") as f:
        content = f.read()

    # Extraer todas las líneas que contienen números hexadecimales
    pattern = re.compile(r"\{([^\}]*)\}")
    matches = pattern.findall(content)

    font_data = []
    for match in matches:
        # extraer los hexadecimales dentro de cada fila
        hex_values = re.findall(r"0x[0-9A-Fa-f]+", match)
        if len(hex_values) == 7:  # cada carácter tiene 7 filas
            pattern_bits = []
            for hex_val in hex_values:
                num = int(hex_val, 16)
                # convertir a 5 bits (columna)
                bits = [(num >> i) & 1 for i in range(5)]
                pattern_bits.extend(bits[::-1])  # invertir para tener orden natural
            font_data.append(pattern_bits)
    return np.array(font_data, dtype=float)

X = load_font_from_header(font_path)
print(X.shape)  # por ejemplo (32, 35)

import matplotlib.pyplot as plt

# plt.imshow(X[1].reshape(7, 5), cmap="gray_r")
# plt.axis("off")
# plt.show()

# X = font_to_binary_matrix(font_path)
# print(X.shape)
# plt.imshow(X[1].reshape(7, 5), cmap="gray_r")
# plt.show()

autoenc = MLP(n_input=35, n_hidden=2, n_output=35, learning_rate=0.001, optimizer='adam')

loss = autoenc.train(X, X, epochs=1000, batch_size=8)

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(loss, label="adam")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de error durante el entrenamiento")
plt.show()

# Z = autoenc.encode(X)

# import matplotlib.pyplot as plt
# plt.scatter(Z[:,0], Z[:,1])
# for i, (x, y) in enumerate(Z):
#     plt.text(x, y, str(i), fontsize=8)
# plt.title("Espacio latente 2D del autoencoder")
# plt.show()