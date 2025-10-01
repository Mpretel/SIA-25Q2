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



# Perceptron Class
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.001):
        # Initialize weights and bias to small random values
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_inputs)]
        self.bias = random.uniform(-0.1, 0.1)
        # Set learning rate
        self.learning_rate = learning_rate
        
    # Activation Function
    def activation_function(self, z, method, beta=1.0):
        if method == 'step':
            tita = 1 if z >= 0 else -1
            tita_prima = 1
        if method == 'linear':
            tita = z
            tita_prima = 1
        if method == 'sigmoid':
            tita = 1 / (1 + np.exp(-2 * beta * z))
            tita_prima = 2 * beta * tita * (1 - tita)
        if method == 'tanh':
            tita = np.tanh(beta * z)
            tita_prima = beta * (1 - tita**2)
        return tita, tita_prima

    def predict(self, x, method, beta=1.0):
        # Calculate the weighted sum
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        # Compute activation given by the activation function
        y_pred, tita_prima = self.activation_function(z, method, beta)
        return y_pred, tita_prima

    def calculate_error(self, yi, y_pred):
        return yi - y_pred
    
    def train(self, X, y, epochs, epsilon=0.0, method='linear', beta=1.0):
        error_history = []

        # For the fixed number of epochs:
        for epoch in range(epochs):
            # For each training example in the dataset
            for xi, yi in zip(X, y):
                # Prediction
                y_pred, tita_prima = self.predict(xi, method, beta)

                # Calculate error: y_real - y_pred
                error = yi - y_pred

                # Update the weights and bias
                # wi = wi + learning_rate * error * tita_prima * xi_j
                self.weights = [w + self.learning_rate * error * tita_prima * xi_j for w, xi_j in zip(self.weights, xi)]
                # bias = bias + learning_rate * tita_prima * error
                self.bias += self.learning_rate * tita_prima * error

            total_error = sum(abs(self.calculate_error(yi, self.predict(xi, method, beta)[0])) for xi, yi in zip(X, y))

            print(f"Epoch {epoch+1} - Errors: {round(total_error, 4)}")
            error_history.append(total_error)

            # Early stopping due to convergence
            if total_error <= epsilon:
                print(f"Converged at epoch {epoch+1} with total_error={round(total_error, 4)}")
                return error_history, epoch+1  # converged early

        return error_history, epochs  # no convergence within limit
    
    
X, y = load_csv("TP3-ej2-conjunto.csv")

#plt_datapoints_3d(X, y)

X_norm, y_norm, y_min, y_max = normalize_data(X, y)




"""
# -------------------------------------------------------------------------
# ENTRENAR UN PERCEPTRON
# -------------------------------------------------------------------------
# Train Perceptron 
method = 'sigmoid'
beta = 5.0
lr = 0.1
epsilon = 0.0
n_epochs = 1000

perceptron = Perceptron(n_inputs=X_norm.shape[1], learning_rate=lr)
perceptron.train(X_norm, y_norm, epochs=n_epochs, epsilon=epsilon, method=method, beta=beta)

# Obtener predicciones
y_pred = [denormalize_y(perceptron.predict(xi, method=method, beta=beta)[0], y_min, y_max) for xi in X_norm]

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
plt.show()

"""



"""
# -------------------------------------------------------------------------
# CAPACIDAD DE APRENDIZAJE 
# -------------------------------------------------------------------------

# GRID SEARCH
# GRID SEARCH con k corridas
methods = ['linear', 'sigmoid', 'tanh']
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

n_epochs = 1000
epsilon = 0.3
k = 10  # cantidad de corridas

results = []

for method in methods:
    for lr in learning_rates:
        if method in ['sigmoid', 'tanh']:
            for beta in betas:
                errors = []
                conv_epochs = []
                for _ in range(k):
                    perceptron = Perceptron(n_inputs=X_norm.shape[1], learning_rate=lr)
                    error_history, conv_epoch = perceptron.train(
                        X_norm, y_norm,
                        epsilon=epsilon, epochs=n_epochs,
                        method=method, beta=beta
                    )
                    y_pred = [denormalize_y(
                                perceptron.predict(xi, method=method, beta=beta)[0],
                                y_min, y_max)
                              for xi in X_norm]
                    err = np.sum(np.abs(y - y_pred))
                    errors.append(err)
                    conv_epochs.append(conv_epoch)

                # promedios
                mean_err = round(np.mean(errors), 1)
                mean_epoch = round(np.mean(conv_epochs), 1)
                results.append((method, lr, beta, mean_err, mean_epoch))

        else:
            errors = []
            conv_epochs = []
            for _ in range(k):
                perceptron = Perceptron(n_inputs=X_norm.shape[1], learning_rate=lr)
                error_history, conv_epoch = perceptron.train(
                    X_norm, y_norm,
                    epsilon=epsilon, epochs=n_epochs,
                    method=method
                )
                y_pred = [denormalize_y(
                            perceptron.predict(xi, method=method)[0],
                            y_min, y_max)
                          for xi in X_norm]
                err = np.sum(np.abs(y - y_pred))
                errors.append(err)
                conv_epochs.append(conv_epoch)

            mean_err = round(np.mean(errors), 1)
            mean_epoch = round(np.mean(conv_epochs), 1)
            results.append((method, lr, None, mean_err, mean_epoch))

# Mostrar resultados ordenados por error promedio
results_sorted = sorted(results, key=lambda x: x[3])
for i, r in enumerate(results_sorted, 1):
    print(f"{i}. {r[0]}, LR: {r[1]}, Beta: {r[2]}, Mean Error: {r[3]}, Mean Epochs: {r[4]}")

"""

"""
# -------------------------------------------------------------------------
# CURVAS DE ERROR DEL MEJOR PERCEPTRON DE CADA METODO SEGUN CAPACIDAD DE APRENDIZAJE
# Linear, LR=0.01
# Sigmoid, LR=0.1, Beta=1.0
# Tanh, LR=0.01, Beta=2.0
# -------------------------------------------------------------------------
best_params = {
    'linear': (0.01, None),
    'sigmoid': (0.1, 1.0),
    'tanh': (0.01, 2.0)
}
# ---------- FIGURA 1: y_real vs y_pred ----------
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

# ---------- FIGURA 2: evolución del error ----------
fig2, ax2 = plt.subplots(figsize=(10, 6))

for i, (method, (lr, beta)) in enumerate(best_params.items()):
    perceptron = Perceptron(n_inputs=X_norm.shape[1], learning_rate=lr)
    error_history, conv_epoch = perceptron.train(
        X_norm, y_norm, epochs=1000, epsilon=0.0, method=method, beta=beta
    )

    y_pred = [denormalize_y(
                perceptron.predict(xi, method=method, beta=beta)[0],
                y_min, y_max) for xi in X_norm]

    # --- Figura 1: scatter y vs y_pred ---
    axes[i].scatter(y, y_pred, c="blue", alpha=0.6, label="Predicciones")
    axes[i].plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="y = x (perfecto)")
    axes[i].set_xlabel("y real")
    axes[i].set_ylabel("y predicho")
    axes[i].set_title(f"{method}")
    axes[i].legend()

    # --- Figura 2: curvas de error ---
    ax2.plot(error_history, label=f"{method} (LR={lr}, Beta={beta})")

# Mostrar figura 1
fig1.suptitle("Comparación y_real vs y_predicho", fontsize=14)
fig1.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste para no tapar el suptitle

# Configurar figura 2
ax2.set_xlabel("Época")
ax2.set_ylabel("Error total")
ax2.set_title("Evolución del error por método")
ax2.legend()
plt.show()
"""


"""
# -------------------------------------------------------------------------
# CAPACIDAD DE GENERALIZACIÓN: elegimos el mejor perceptron --> hacemos un simple train test split
# Sigmoid, LR=0.1, Beta=1.0
# -------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.3)

# Train Perceptron SOLO con train
method = 'sigmoid' 
lr = 0.1
beta = 1.0
epochs = 1000
epsilon = 0.0
perceptron = Perceptron(n_inputs=X_norm.shape[1], learning_rate=lr)
perceptron.train(X_train, y_train, epochs=epochs, epsilon=epsilon, method=method, beta=beta)

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
plt.show()

# --- Predicciones en TEST ---
y_test_pred = [perceptron.predict(xi, method=method)[0] for xi in X_test]
y_test_pred = denormalize_y(np.array(y_test_pred), y_min, y_max)
y_test_real = denormalize_y(y_test, y_min, y_max)

# --- Métricas ---
mse_train = np.mean((y_train_real - y_train_pred) ** 2)/len(y_train_real)
mse_test = np.mean((y_test_real - y_test_pred) ** 2)/len(y_test_real)
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
plt.show()

"""

# -------------------------------------------------------------------------
# CAPACIDAD DE GENERALIZACIÓN: elegimos el mejor perceptron --> hacemos k fold cross-validation
# Sigmoid, LR=0.1, Beta=1.0
# -------------------------------------------------------------------------


def k_fold_split(X, y, k=5, seed=42):
    """Genera índices de train/test para K-Fold cross-validation"""
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        current = stop
    return folds


def cross_validate_perceptron(X, y, k=5, method='sigmoid', lr=0.1, beta=1.0, epochs=1000, epsilon=0.0):
    folds = k_fold_split(X, y, k)
    mse_train_list = []
    mse_test_list = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        perceptron = Perceptron(n_inputs=X.shape[1], learning_rate=lr)
        perceptron.train(X_train, y_train, epochs=epochs, epsilon=epsilon, method=method, beta=beta)

        # Predicciones
        y_train_pred = [perceptron.predict(xi, method, beta)[0] for xi in X_train]
        y_test_pred  = [perceptron.predict(xi, method, beta)[0] for xi in X_test]

        # Denormalizar
        y_train_pred = denormalize_y(np.array(y_train_pred), y_min, y_max)
        y_test_pred  = denormalize_y(np.array(y_test_pred), y_min, y_max)
        y_train_real = denormalize_y(y_train, y_min, y_max)
        y_test_real  = denormalize_y(y_test, y_min, y_max)

        # Calcular MSE
        mse_train = np.mean((y_train_real - y_train_pred)**2)/len(y_train_real)
        mse_test  = np.mean((y_test_real - y_test_pred)**2)/len(y_test_real)

        print("len train set:", len(y_train_real))
        print("len test set:", len(y_test_real))

        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)

        print(f"Fold {fold+1}: MSE Train = {mse_train:.4f}, MSE Test = {mse_test:.4f}")

        # --- Gráfico ---
        plt.figure(figsize=(6,6))
        plt.scatter(y_train_real, y_train_pred, c="blue", alpha=0.6, label="Predicciones (train)")
        plt.scatter(y_test_real, y_test_pred, c="green", alpha=0.6, label="Predicciones (test)")
        plt.plot([y_min, y_max], [y_min, y_max], "r--", label="y = x (perfecto)")
        plt.xlabel("y real")
        plt.ylabel("y predicho")
        plt.title(f"Fold {fold+1}")
        plt.legend()
        plt.show()

    print("\n--- Resumen K-Fold ---")
    print(f"MSE Train promedio: {np.mean(mse_train_list):.4f} ± {np.std(mse_train_list):.4f}")
    print(f"MSE Test promedio: {np.mean(mse_test_list):.4f} ± {np.std(mse_test_list):.4f}")

    return mse_train_list, mse_test_list


mse_train_cv, mse_test_cv = cross_validate_perceptron(
    X_norm, y_norm, k=5, method='sigmoid', lr=0.1, beta=1.0, epochs=1000
)
