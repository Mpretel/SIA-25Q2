import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from perceptrons import MLP

SEED = 42
np.random.seed(SEED)

def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = len(labels) if labels is not None else max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm



colors_optimizers = {'gd': 'tab:green', 'momentum': 'tab:orange', 'adam': 'tab:blue'}



# ------------------------------
# Funcion XOR
# ------------------------------

# Datos XOR
X = np.array([[1, -1, 1],
              [1, 1, -1],
              [1, -1, -1],
              [1, 1, 1]])
y = np.array([[1],
              [1],
              [-1],
              [-1]])

# Crear MLP
optimizer = 'momentum'
mlp = MLP(n_input=3, n_hidden=3, n_output=1, learning_rate=0.1, activation_function='tanh', optimizer=optimizer)

# Entrenar
loss_history = mlp.train(X, y, epochs=1000, epsilon=0.0, batch_size=4)

# Predicciones finales
final_output = mlp.forward(X)
predictions = mlp.predict(X)

print(f"Predicciones continuas ({mlp.activation_function_name}, {mlp.optimizer}):")
print(final_output)
print("Predicciones finales (-1/1):")
print(predictions)
print("Salida esperada:")
print(y)

# plot loss history
plt.plot(loss_history, color=colors_optimizers[optimizer])
plt.xlabel("Epoch") 
plt.ylabel("Loss")
plt.title(f"Curva de error durante el entrenamiento ({mlp.activation_function_name}, {mlp.optimizer})")
plt.show()

# Lista de optimizadores
optimizers = ['gd', 'momentum', 'adam']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8,5))

for opt in (optimizers):
    # Crear MLP con el optimizador actual
    mlp = MLP(n_input=3, n_hidden=3, n_output=1,
              learning_rate=0.1,
              activation_function='tanh',
              optimizer=opt)
    
    # Entrenar
    loss_history = mlp.train(X, y, epochs=1000, epsilon=0.0, batch_size=4)
    
    # Graficar curva de error
    plt.plot(loss_history, color=colors_optimizers[opt], label=opt.upper())

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Comparación de evolución del error (GD vs Momentum vs Adam)")
plt.legend()
plt.show()



"""
# ------------------------------------------------------------------------------------------
# DIGITOS
# ------------------------------------------------------------------------------------------
filename = "TP3-ej3-digitos.txt"
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, filename)

# Leer archivo
data = np.loadtxt(file_path)

n_rows, n_cols = data.shape  # (70, 5)
digit_height = 7
digit_width = 5
digit_size = digit_height * digit_width

# Número de dígitos en el archivo
n_digits = n_rows // digit_height

# Vectorizo cada dígito (7x5 -> 35)
X = data.reshape(n_digits, digit_height, digit_width)   # (10, 7, 5)
X = X.reshape(n_digits, digit_size)                     # (10, 35)
"""

"""
# ------------------------------
# Discriminar paridad
# ------------------------------
optimizer = 'gd'

# Etiquetas: pares = 1, impares = 0
y = np.array([[1 if d % 2 == 0 else 0] for d in range(n_digits)])

# Crear y entrenar MLP
mlp = MLP(n_input=35, n_hidden=10, n_output=1, learning_rate=0.1, activation_function='sigmoid', optimizer=optimizer)
loss_history = mlp.train(X, y, epochs=700, epsilon=0.0, batch_size=1)

y_pred = mlp.predict(X)
print("Esperado:", y.flatten())
print("Predicho:", y_pred.flatten())

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label=optimizer, color=colors_optimizers[optimizer])
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de error durante el entrenamiento")
plt.show()

# ------------------------------------------
# Calcular métricas de clasificación
# Calcular accuracy
acc = accuracy_score(y, y_pred)
print(f"\nAccuracy: {acc:.2f}")

# Calcular matriz de confusión absoluta
labels = [0, 1]  # 0 = impar, 1 = par
cm_abs = confusion_matrix(y, y_pred, labels=labels)
print("\nMatriz de confusión:")
print(cm_abs)

# Matriz de confusión relativa (normalizada por fila)
cm_rel = cm_abs.astype(float)
cm_rel = ((cm_rel / cm_rel.sum(axis=1, keepdims=True))*100).astype(int)

# Graficar ambas matrices lado a lado
fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(cm_abs, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f"Absoluta")
axes[0].set_xlabel("Predicho")
axes[0].set_ylabel("Esperado")

sns.heatmap(cm_rel, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=100, ax=axes[1])
axes[1].set_title(f"Relativa (%)")
axes[1].set_xlabel("Predicho")
axes[1].set_ylabel("Esperado")

plt.tight_layout()
plt.show()



# Comparación de optimizadores
batch_sizes = [1, 10]
learning_rates = [0.1, 0.01]
optimizers = ['adam', 'momentum', 'gd']

for batch_size in batch_sizes:
    for lr in learning_rates:
        histories = {}
        for opt in optimizers:
            mlp = MLP(
                n_input=35,
                n_hidden=10,
                n_output=1,
                learning_rate=lr,
                activation_function='sigmoid',
                optimizer=opt
            )

            loss_history = mlp.train(X, y, epochs=700, epsilon=0.0, batch_size=batch_size)
            histories[opt] = loss_history

            y_pred = mlp.predict(X)

            print("Esperado:", y)
            print("Predicho:", y_pred)

            # ------------------------------------------
            # Calcular métricas de clasificación
            # Calcular accuracy
            acc = accuracy_score(y, y_pred)
            print(f"\nAccuracy: {acc:.2f}")

            # Calcular matriz de confusión absoluta
            labels = [0, 1]  # 0 = impar, 1 = par
            cm_abs = confusion_matrix(y, y_pred, labels=labels)
            print("\nMatriz de confusión:")
            print(cm_abs)

            # Matriz de confusión relativa (normalizada por fila)
            cm_rel = cm_abs.astype(float)
            cm_rel = ((cm_rel / cm_rel.sum(axis=1, keepdims=True))*100).astype(int)

            # Graficar ambas matrices lado a lado
            fig, axes = plt.subplots(1, 2, figsize=(12,5))

            sns.heatmap(cm_abs, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title(f"Absoluta - lr {lr}, batch size {batch_size}, {opt}")
            axes[0].set_xlabel("Predicho")
            axes[0].set_ylabel("Esperado")

            sns.heatmap(cm_rel, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=100, ax=axes[1])
            axes[1].set_title(f"Relativa (%) - lr {lr}, batch size {batch_size}, {opt}")
            axes[1].set_xlabel("Predicho")
            axes[1].set_ylabel("Esperado")

            plt.tight_layout()
            plt.show()

        plt.figure(figsize=(8, 5))
        for opt in optimizers:
            plt.plot(histories[opt], label=opt)

        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title(f"Comparación de optimizadores (lr = {lr}, batch size = {batch_size})")
        plt.legend()
        plt.show()

"""
"""
# ------------------------------
# Discriminar digitos
# ------------------------------
# Etiquetas: numeros 0-9
y = np.array([[1,0,0,0,0,0,0,0,0,0], 
              [0,1,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,1]])
y_label = np.argmax(y, axis=1)
"""

"""
# Comparación de optimizadores
learning_rates = [0.1, 0.01]
batch_sizes = [1, 10]
optimizers = ['adam', 'momentum', 'gd']
histories = {}

for lr in learning_rates:
    for batch_size in batch_sizes:
        histories = {}
        for opt in optimizers:
            print(f"\nEntrenando con optimizador: {opt}")
            mlp = MLP(
                n_input=35,
                n_hidden=10,
                n_output=10,
                learning_rate=lr,
                activation_function='sigmoid',
                optimizer=opt
            )

            loss_history = mlp.train(X, y, epochs=1000, epsilon=0.0, batch_size=1)
            histories[opt] = loss_history

            y_pred_label = mlp.predict(X, method='multiclass')

            print("Esperado:", y_label)
            print("Predicho:", y_pred_label)


            # ------------------------------------------
            # Calcular métricas de clasificación
            # Calcular accuracy
            acc = accuracy_score(y_label, y_pred_label)
            print(f"\nAccuracy: {acc:.2f}")

            # Calcular matriz de confusión absoluta
            cm_abs = confusion_matrix(y_label, y_pred_label, labels=list(range(10)))
            print("\nMatriz de confusión:")
            print(cm_abs)

            # Matriz de confusión relativa (normalizada por fila)
            cm_rel = cm_abs.astype(float)
            cm_rel = ((cm_rel / cm_rel.sum(axis=1, keepdims=True))*100).astype(int)

            # Graficar ambas matrices lado a lado
            fig, axes = plt.subplots(1, 2, figsize=(12,5))

            sns.heatmap(cm_abs, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title(f"Absoluta - lr {lr}, batch {batch_size}, {opt}")
            axes[0].set_xlabel("Predicho")
            axes[0].set_ylabel("Esperado")

            sns.heatmap(cm_rel, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=100, ax=axes[1])
            axes[1].set_title(f"Relativa (%) - lr {lr}, batch {batch_size}, {opt}")
            axes[1].set_xlabel("Predicho")
            axes[1].set_ylabel("Esperado")

            plt.tight_layout()
            plt.show()
            
        plt.figure(figsize=(8, 5))
        for opt in optimizers:
            plt.plot(histories[opt], label=opt, color=colors_optimizers[opt])

        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title(f"Comparación de optimizadores (lr = {lr})")
        plt.legend()
        plt.show()
"""

"""
# ------------------------------
# Agregamos Ruido
# ------------------------------


optimizer = 'gd'

# Crear y entrenar MLP
mlp = MLP(n_input=35, n_hidden=10, n_output=10, learning_rate=0.1, activation_function='sigmoid', optimizer=optimizer)
loss_history = mlp.train(X, y, epochs=1000, epsilon=0.0, batch_size=1)

y_pred_label = mlp.predict(X, method='multiclass')

print("Esperado:", y_label)
print("Predicho:", y_pred_label)

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label=optimizer, color=colors_optimizers[optimizer])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de error durante el entrenamiento")
plt.legend()
plt.show()




def add_noise(X, noise_level=0.1):
    noisy_X = X.copy()
    flip_mask = np.random.rand(*X.shape) < noise_level
    noisy_X[flip_mask] = 1 - noisy_X[flip_mask]
    return noisy_X


noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

fig, axes = plt.subplots(len(noise_levels), 10, figsize=(12, 2*len(noise_levels)))
for i, nl in enumerate(noise_levels):
    X_noisy = add_noise(X, noise_level=nl)
    y_pred_noisy_label = mlp.predict(X_noisy, method='multiclass')
    
    for j in range(10):
        axes[i, j].imshow(
            X_noisy[j].reshape(digit_height, digit_width),
            cmap='gray_r',  # inverso: 0=blanco, 1=negro
            vmin=0, vmax=1
        )
        axes[i, j].axis('off')

plt.suptitle("Imágenes de los números con distintos niveles de ruido")
plt.show()


for nl in noise_levels:
    X_noisy = add_noise(X, noise_level=nl)
    y_pred_noisy_label = mlp.predict(X_noisy, method='multiclass')

    acc = accuracy_score(y_label, y_pred_noisy_label)
    print(f"Ruido: {nl*100:.1f}% - Exactitud: {acc*100:.1f}%")
    
    # Matriz de confusión
    cm = confusion_matrix(y_label, y_pred_noisy_label, labels=list(range(10)))
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de confusión - Ruido {nl*100:.1f}%")
    plt.xlabel("Predicho")
    plt.ylabel("Esperado")
    plt.show()




n_variants = 10  # número de versiones ruidosas por dígito
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

for nl in noise_levels:
    all_preds = []
    all_labels = []

    for _ in range(n_variants):
        X_noisy = add_noise(X, noise_level=nl)
        y_pred_noisy_label = mlp.predict(X_noisy, method='multiclass')

        all_preds.extend(y_pred_noisy_label)
        all_labels.extend(y_label)  # repetimos las etiquetas reales

    # Exactitud promedio
    acc = accuracy_score(all_labels, all_preds)
    print(f"Ruido: {nl*100:.1f}% - Exactitud promedio: {acc*100:.1f}%")

    # Matriz de confusión absoluta
    cm_abs = confusion_matrix(all_labels, all_preds, labels=list(range(10)))

    # Matriz de confusión relativa (normalizada por fila)
    cm_rel = cm_abs.astype(float)
    cm_rel = ((cm_rel / cm_rel.sum(axis=1, keepdims=True))*100).astype(int)

    # Graficar ambas matrices lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    
    sns.heatmap(cm_abs, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f"Absoluta - Ruido {nl*100:.1f}%")
    axes[0].set_xlabel("Predicho")
    axes[0].set_ylabel("Esperado")
    
    sns.heatmap(cm_rel, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=100, ax=axes[1])
    axes[1].set_title(f"Relativa - Ruido {nl*100:.1f}%")
    axes[1].set_xlabel("Predicho")
    axes[1].set_ylabel("Esperado")
    
    plt.tight_layout()
    plt.show()
"""