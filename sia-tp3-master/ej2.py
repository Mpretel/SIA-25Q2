import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D



# Read CSV
def load_csv(filename):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, filename)
    df = pd.read_csv(file_path)
    # X = x1, x2, x3
    X = df.iloc[:, :-1].values
    # y = last column
    y = df.iloc[:, -1].values
    return X, y

def plt_datapoints_3d(X, y):
    # Plot data in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", s=60)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title("Puntos en 3D coloreados por y")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=10) # color bar for y
    cbar.set_label("y")
    plt.show()


def normalize_data(X, y):
    # Normalize X and y to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min)
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    y = (y - y_min) / (y_max - y_min)
    return X, y, y_min, y_max


def denormalize_y(y, y_min, y_max):
    return y * (y_max - y_min) + y_min


# Train/test split
def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    test_size = int(n_samples * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Activation Function
def activation_function(z, method='step'):
    if method == 'step':
        tita = 1 if z >= 0 else -1
        tita_prima = 1
    if method == 'linear':
        tita = z
        tita_prima = 1
    if method == 'sigmoid':
        beta = 1.0
        tita = 1 / (1 + np.exp(-2 * beta * z))
        tita_prima = 2 * beta * tita * (1 - tita)
    if method == 'tanh':
        beta = 1.0
        tita = np.tanh(beta * z)
        tita_prima = beta * (1 - tita**2)
    return tita, tita_prima

# Perceptron Class
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.001):
        # Initialize weights and bias to small random values
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_inputs)]
        self.bias = random.uniform(-0.1, 0.1)
        # Set learning rate
        self.learning_rate = learning_rate

    def predict(self, x, method):
        # Calculate the weighted sum
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        # Compute activation given by the activation function
        y_pred, tita_prima = activation_function(z, method)
        return y_pred, tita_prima

    def train(self, X, y, epochs, epsilon=0.0, method='linear'):
        # For the fixed number of epochs:
        for epoch in range(epochs):
            total_error = 0
            # For each training example in the dataset
            for xi, yi in zip(X, y):
                # Prediction
                y_pred, tita_prima = self.predict(xi, method)

                # Calculate error: y_real - y_pred
                error = yi - y_pred
                total_error += abs(error)

                # Update the weights and bias
                # wi = wi + learning_rate * error * tita_prima * xi_j
                self.weights = [w + self.learning_rate * error * tita_prima * xi_j for w, xi_j in zip(self.weights, xi)]
                # bias = bias + learning_rate * tita_prima * error
                self.bias += self.learning_rate * tita_prima * error

            print(f"Epoch {epoch+1} - Errors: {round(total_error, 4)}")

            # Early stopping due to convergence
            if total_error <= epsilon:
                print(f"Converged at epoch {epoch+1} with total_error={round(total_error, 4)}")
                break


X, y = load_csv("TP3-ej2-conjunto.csv")

#plt_datapoints_3d(X, y)

X_norm, y_norm, y_min, y_max = normalize_data(X, y)


# -------------------------------------------------------------------------
# CAPACIDAD DE APRENDIZAJE 
# -------------------------------------------------------------------------

# Train Perceptron 
method = 'sigmoid'  # 'step', 'linear', 'sigmoid', 'tanh'
perceptron = Perceptron(n_inputs=X_norm.shape[1], learning_rate=0.01)
perceptron.train(X_norm, y_norm, epochs=1000, epsilon=0.0, method=method)

# Obtener predicciones
y_pred = [denormalize_y(perceptron.predict(xi, method=method)[0], y_min, y_max) for xi in X_norm]

for yi, yi_pred in zip(y, y_pred):
    print(f"{yi} -> {round(yi_pred, 4)}")

# Graficar
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, c="blue", alpha=0.6, label="Predicciones")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="y = x (perfecto)")
plt.xlabel("y real")
plt.ylabel("y predicho")
plt.title("Comparación y real vs y predicho")
plt.legend()
plt.grid(True)
plt.show()



# -------------------------------------------------------------------------
# CAPACIDAD DE GENERALIZACIÓN 
# -------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm)



# Train Perceptron SOLO con train
method = 'sigmoid'  # 'step', 'linear', 'sigmoid', 'tanh'
perceptron = Perceptron(n_inputs=X_norm.shape[1], learning_rate=0.01)
perceptron.train(X_train, y_train, epochs=1000, epsilon=0.0, method=method)

# --- Predicciones en TRAIN ---
y_train_pred = [perceptron.predict(xi, method=method)[0] for xi in X_train]
y_train_pred = denormalize_y(np.array(y_train_pred), y_min, y_max)
y_train_real = denormalize_y(y_train, y_min, y_max)

# --- Gráfico comparación en TEST ---
plt.figure(figsize=(6,6))
plt.scatter(y_train_real, y_train_pred, c="blue", alpha=0.6, label="Predicciones (train)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="y = x (perfecto)")
plt.xlabel("y real")
plt.ylabel("y predicho")
plt.title("Generalización del perceptrón (TRAIN)")
plt.legend()
plt.grid(True)
plt.show()

# --- Predicciones en TEST ---
y_test_pred = [perceptron.predict(xi, method=method)[0] for xi in X_test]
y_test_pred = denormalize_y(np.array(y_test_pred), y_min, y_max)
y_test_real = denormalize_y(y_test, y_min, y_max)

# --- Métricas ---
mse_train = np.mean((y_train_real - y_train_pred) ** 2)
mse_test = np.mean((y_test_real - y_test_pred) ** 2)
print(f"MSE Train: {mse_train:.4f}")
print(f"MSE Test: {mse_test:.4f}")

# --- Gráfico comparación en TEST ---
plt.figure(figsize=(6,6))
plt.scatter(y_test_real, y_test_pred, c="blue", alpha=0.6, label="Predicciones (test)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="y = x (perfecto)")
plt.xlabel("y real")
plt.ylabel("y predicho")
plt.title("Generalización del perceptrón (TEST)")
plt.legend()
plt.grid(True)
plt.show()
