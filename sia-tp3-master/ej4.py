from ej3 import MLP
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

dict = {0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

def one_hot_encode(y):
    return np.array([dict[label] for label in y.flatten()])

# Cargar los datos
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Aplanar cada imagen 28x28 a un vector de 784
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalizar los datos para sigmoid
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode las etiquetas
y_train_onehot = one_hot_encode(y_train.reshape(-1, 1))
y_test_onehot = one_hot_encode(y_test.reshape(-1, 1))

"""
mlp = MLP(
    n_input=784,
    n_hidden=64,    # por ejemplo
    n_output=10,
    learning_rate=0.1,
    activation_function='sigmoid',
    optimizer='momentum'
)

loss_history = mlp.train(X_train, y_train_onehot, epochs=50, batch_size=32)
y_pred_label = mlp.predict(X_test, method='multiclass')

# Accuracy
accuracy = np.mean(y_pred_label == y_test)
print(f"Accuracy en test: {accuracy*100:.2f}%")

# Mostrar comparación de algunos ejemplos
print("Esperado:", y_test[:10])
print("Predicho:", y_pred_label[:10])
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(loss_history, color="orange", label="momentum")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de error durante el entrenamiento (MNIST)")
plt.legend()
plt.show()
"""

# Comparación de optimizadores

optimizers = ['adam', 'momentum', 'gd']
histories = {}

for opt in optimizers:
    print(f"\nEntrenando con optimizador: {opt}")
    mlp = MLP(
        n_input=784,
        n_hidden=64,
        n_output=10,
        learning_rate=0.1,
        activation_function='sigmoid',
        optimizer=opt
    )

    loss_history = mlp.train(X_train, y_train_onehot, epochs=100, epsilon=0.0, batch_size=100)
    histories[opt] = loss_history

    y_pred_label = mlp.predict(X_test, method='multiclass')

    # Accuracy
    accuracy = np.mean(y_pred_label == y_test)
    print(f"Accuracy en test: {accuracy*100:.2f}%")

    print("Esperado:", y_test[:10])
    print("Predicho:", y_pred_label[:10])

plt.figure(figsize=(8, 5))
for opt in optimizers:
    plt.plot(histories[opt], label=opt)

plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Comparación de optimizadores (lr = 0.1)")
plt.legend()
plt.grid(True)
plt.show()
