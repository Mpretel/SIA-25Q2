from perceptrons import MLP
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import seaborn as sns

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
# Calcular proporciones

unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

prop_train = counts_train / len(y_train)
prop_test = counts_test / len(y_test)

# Gráfico
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axes[0].bar(unique_train, prop_train, color='steelblue')
axes[0].set_title("Proporciones en Train")
axes[0].set_xlabel("Dígito")
axes[0].set_ylabel("Proporción")

axes[1].bar(unique_test, prop_test, color='orange')
axes[1].set_title("Proporciones en Test")
axes[1].set_xlabel("Dígito")

plt.tight_layout()
plt.show()
"""

mlp = MLP(
    n_input=784,
    n_hidden=64,    
    n_output=10,
    learning_rate=0.1,
    activation_function='sigmoid',
    optimizer='gd'
)

loss_history = mlp.train(X_train, y_train_onehot, epochs=50, batch_size=100)
y_pred_label = mlp.predict(X_test, method='multiclass')

# Accuracy
accuracy = np.mean(y_pred_label == y_test)
print(f"Accuracy en test: {accuracy*100:.2f}%")

# Mostrar comparación de algunos ejemplos
print("Esperado:", y_test[:10])
print("Predicho:", y_pred_label[:10])
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(loss_history, color="orange", label="gd")
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
        learning_rate=0.01,
        activation_function='sigmoid',
        optimizer=opt
    )

    loss_history = mlp.train(X_train, y_train_onehot, epochs=300, epsilon=0.0, batch_size=10000)
    histories[opt] = loss_history

    y_pred_label = mlp.predict(X_train, method='multiclass')

    # Accuracy en Train
    accuracy = np.mean(y_pred_label == y_train)
    print(f"Accuracy en train: {accuracy*100:.2f}%")

    # # Accuracy en Test
    # accuracy = np.mean(y_pred_label == y_test)
    # print(f"Accuracy en test: {accuracy*100:.2f}%")

    print("Esperado:", y_train[:10])
    print("Predicho:", y_pred_label[:10])

plt.figure(figsize=(8, 5))
for opt in optimizers:
    plt.plot(histories[opt], label=opt)

plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Comparación de optimizadores (lr = 0.01)")
plt.legend()
plt.grid(True)
plt.show()
"""

"""


optimizers = ['gd']
#optimizers = ['adam', 'momentum', 'gd']
histories = {}
y_test_onehot_label = np.argmax(y_test_onehot, axis=1)

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

    loss_history = mlp.train(X_train, y_train_onehot, epochs=500, epsilon=0.0, batch_size=100)
    histories[opt] = loss_history

    y_pred_label = mlp.predict(X_test, method='multiclass')

    acc = accuracy_score(y_test_onehot_label, y_pred_label)
    print(f"Optimizador: {opt} - Exactitud: {acc*100:.1f}%")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test_onehot_label, y_pred_label, labels=list(range(10)))
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de confusión - Optimizador {opt}")
    plt.xlabel("Predicho")
    plt.ylabel("Esperado")
    plt.show()

"""