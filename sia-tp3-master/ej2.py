import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D


SEED = 42
np.random.seed(SEED)
random.seed(SEED)



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

# Train/test split
def train_test_split(X, y, test_size=0.2):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    test_size = int(n_samples * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def k_fold_split(X, k=5):
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


# Perceptron Class
class Perceptron:
    def __init__(self, n_inputs, X_min, X_max, y_min, y_max, activation_function='linear', beta=1.0, learning_rate=0.001):
        # Initialize weights and bias to small random values
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_inputs)]
        self.bias = random.uniform(-0.1, 0.1)
        # Set learning rate
        self.learning_rate = learning_rate
        # Set activation function parameters
        self.activation_function_name = activation_function
        self.beta = beta

        # Store min and max for normalization/denormalization
        self.X_min, self.X_max, self.y_min, self.y_max = X_min, X_max, y_min, y_max

    # Normalize and denormalize methods
    def normalize_x(self, x):
        if self.activation_function_name == 'sigmoid':
            xn = (x - self.X_min) / (self.X_max - self.X_min)
        else:  # linear o tanh
            xn = 2 * (x - self.X_min) / (self.X_max - self.X_min) - 1
        return xn
    
    def normalize_y(self, y):
        if self.activation_function_name == 'sigmoid':
            yn = (y - self.y_min) / (self.y_max - self.y_min)
        else:  # linear o tanh
            yn = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        return yn

    def denormalize_y(self, y):
        if self.activation_function_name == 'sigmoid':
            return y * (self.y_max - self.y_min) + self.y_min
        else:
            return ((y + 1) / 2) * (self.y_max - self.y_min) + self.y_min
    
    # Activation Function
    def activation_function(self, z):
        if self.activation_function_name == 'step':
            tita = 1 if z >= 0 else -1
            tita_prima = 1
        if self.activation_function_name == 'linear':
            tita = z
            tita_prima = 1
        if self.activation_function_name == 'sigmoid':
            tita = 1 / (1 + np.exp(-2 * self.beta * z))
            tita_prima = 2 * self.beta * tita * (1 - tita)
        if self.activation_function_name == 'tanh':
            tita = np.tanh(self.beta * z)
            tita_prima = self.beta * (1 - tita**2)
        return tita, tita_prima
    
    # Predict
    def predict(self, x, denormalize=True):
        x = self.normalize_x(x)
        # Calculate the weighted sum
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        # Compute activation given by the activation function
        y_pred, tita_prima = self.activation_function(z)
        if denormalize:
            y_pred = self.denormalize_y(y_pred)
        return y_pred, tita_prima

    # Calculate squared error
    def calculate_error(self, yi, y_pred):
        return (yi - y_pred)**2
    
    # Train the perceptron
    def train(self, X, y, epochs, epsilon=0.0):
        denormalized_error_history = []

        # For the fixed number of epochs:
        for epoch in range(epochs):
            # For each training example in the dataset
            for xi, yi in zip(X, y):
                # Prediction
                yn_pred, tita_prima = self.predict(xi, denormalize=False)

                # Calculate error: y_real - y_pred
                yni = self.normalize_y(yi)
                error = yni - yn_pred

                # Update the weights and bias
                # wi = wi + learning_rate * error * tita_prima * xi_j
                self.weights = [w + self.learning_rate * error * tita_prima * xi_j for w, xi_j in zip(self.weights, xi)]
                # bias = bias + learning_rate * tita_prima * error
                self.bias += self.learning_rate * tita_prima * error

            # Total error for the epoch (both normalized and denormalized)
            denormalized_total_error = 0
            for xi, yi in zip(X, y):
                yi_pred, _ = self.predict(xi, denormalize=True)
                denormalized_total_error += self.calculate_error(yi, yi_pred)
            denormalized_error_history.append(denormalized_total_error)
            print(f"Epoch {epoch+1} - Errors: {round(denormalized_total_error, 4)}")

            # Early stopping due to convergence
            if denormalized_total_error <= epsilon:
                print(f"Converged at epoch {epoch+1} with total_error={round(denormalized_total_error, 4)}")
                return denormalized_error_history, epoch+1  # converged early

        return denormalized_error_history, epochs  # no convergence within limit
    
    
X, y = load_csv("TP3-ej2-conjunto.csv")

X_min, X_max = X.min(axis=0), X.max(axis=0)
y_min, y_max = y.min(), y.max()

#plt_datapoints_3d(X, y)



"""
# -------------------------------------------------------------------------
# ENTRENAR UN PERCEPTRON
# -------------------------------------------------------------------------
# Train Perceptron 
activation_function = 'tanh'
beta = 2.0
lr = 0.01
epsilon = 0.0
n_epochs = 300

perceptron = Perceptron(n_inputs=X.shape[1], X_min=X_min, X_max=X_max, y_min=y_min, y_max=y_max, learning_rate=lr, activation_function=activation_function, beta=beta)
error_history, _ = perceptron.train(X, y, epochs=n_epochs, epsilon=epsilon)

# Obtener predicciones
y_pred = [perceptron.predict(xi)[0] for xi in X]

for yi, yi_pred in zip(y, y_pred):
    print(f"{yi} -> {round(yi_pred, 4)}")

# Graficar
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, c="blue", alpha=0.6, label="Predicciones")
plt.plot([y_min, y_max], [y_min, y_max], "r--", label="y = x (perfecto)")
plt.xlabel("y real")
plt.ylabel("y predicho")
plt.title("Comparación y real vs y predicho")
plt.legend()
plt.show()

# Graficar el error
plt.plot(error_history, label=f"{activation_function} (LR={lr}, Beta={beta})")
plt.xlabel("Epoch")
plt.ylabel("Error total")
plt.title("Curva de error durante el entrenamiento")
plt.legend()
plt.show()
"""




# -------------------------------------------------------------------------
# CAPACIDAD DE APRENDIZAJE 
# -------------------------------------------------------------------------

# GRID SEARCH
# GRID SEARCH con k corridas
activation_functions = ['linear', 'sigmoid', 'tanh']
learning_rates_l = [1E-4, 5E-4, 1E-3, 5E-3, 0.01, 0.05]
learning_rates_nl = [0.01, 0.05, 0.1, 0.2, 0.5]
betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

n_epochs = 1000
epsilon = 0.0
k = 10  # cantidad de corridas

results = []

for activation_function in activation_functions:
    for lr in learning_rates_nl:
        if activation_function in ['sigmoid', 'tanh']:
            for beta in betas:
                errors = []
                conv_epochs = []
                for _ in range(k):
                    perceptron = Perceptron(n_inputs=X.shape[1], X_min=X_min, X_max=X_max, y_min=y_min, y_max=y_max, learning_rate=lr, activation_function=activation_function, beta=beta)
                    error_history, conv_epoch = perceptron.train(X, y, epochs=n_epochs, epsilon=epsilon)
                    y_pred = [perceptron.predict(xi)[0] for xi in X]
                    err = np.sum((y - y_pred)**2)
                    errors.append(err)
                    conv_epochs.append(conv_epoch)

                mean_err = round(np.mean(errors), 1)
                mean_epoch = round(np.mean(conv_epochs), 1)
                results.append((activation_function, lr, beta, mean_err, mean_epoch))

    for lr in learning_rates_l:
        if activation_function == "linear":
            errors = []
            conv_epochs = []
            for _ in range(k):
                perceptron = Perceptron(n_inputs=X.shape[1], X_min=X_min, X_max=X_max, y_min=y_min, y_max=y_max, learning_rate=lr, activation_function=activation_function)
                error_history, conv_epoch = perceptron.train(X, y, epochs=n_epochs, epsilon=epsilon)
                y_pred = [perceptron.predict(xi)[0] for xi in X]
                err = np.sum((y - y_pred)**2)
                errors.append(err)
                conv_epochs.append(conv_epoch)

            mean_err = round(np.mean(errors), 1)
            mean_epoch = round(np.mean(conv_epochs), 1)
            results.append((activation_function, lr, None, mean_err, mean_epoch))

# Mostrar resultados ordenados por error promedio
results_sorted = sorted(results, key=lambda x: x[3])
for i, r in enumerate(results_sorted, 1):
    print(f"{i}. {r[0]}, LR: {r[1]}, Beta: {r[2]}, Mean Error: {r[3]}")


# HEATMAPS DE ERROR SEGUN CAPACIDAD DE APRENDIZAJE
import seaborn as sns

# Pasar resultados a DataFrame
df = pd.DataFrame(results, columns=["method", "lr", "beta", "mean_err", "mean_epoch"])

# Extraer valores globales de min y max para escalar todos los mapas igual
vmin = df["mean_err"].min()
vmax = df["mean_err"].max()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, method in zip(axes, ["linear"]):
    if method in ["linear"]:
        pivot = df[df["method"] == method].copy()
        pivot["beta"] = 0  
        pivot = pivot.pivot(index="beta", columns="lr", values="mean_err")
        a = True

    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap="viridis", vmin=vmin, vmax=vmax,
        ax=ax
    )
    ax.set_title(f"{method}")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Beta")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, method in zip(axes, ["sigmoid", "tanh"]):
    if method in ["sigmoid", "tanh"]:
        # Construir matriz LR vs Beta
        pivot = df[df["method"] == method].pivot(index="beta", columns="lr", values="mean_err")
        a = False
    # else:  # linear no depende de beta → repetimos para que el mapa tenga "forma"
    #     pivot = df[df["method"] == method].copy()
    #     pivot["beta"] = 0  
    #     pivot = pivot.pivot(index="beta", columns="lr", values="mean_err")
    #     a = True

    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap="viridis", vmin=vmin, vmax=vmax,
        #cbar=a,  # cada heatmap con su barra de color
        ax=ax
    )
    ax.set_title(f"{method}")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Beta")

plt.tight_layout()
plt.show()




# -------------------------------------------------------------------------
# CURVAS DE ERROR DEL MEJOR PERCEPTRON DE CADA METODO SEGUN CAPACIDAD DE APRENDIZAJE
# Linear, LR=0.01
# Sigmoid, LR=0.5, Beta=0.5
# Tanh, LR=0.05, Beta=0.5
# -------------------------------------------------------------------------
best_params = {
    'sigmoid1': ('sigmoid', 0.01, 2.0),
    'sigmoid2': ('sigmoid', 0.05, 1.0),
    'sigmoid3': ('sigmoid', 0.1, 1.0),
    'sigmoid4': ('sigmoid', 0.2, 0.5),
    'sigmoid5': ('sigmoid', 0.5, 0.5)
}
best_params = {
    'tanh1': ('tanh', 0.01, 1.0),
    'tanh2': ('tanh', 0.05, 0.5),
    'tanh3': ('tanh', 0.5, 0.1)
}

best_params = {
    'linear': ('linear', 0.001, None),
    'sigmoid': ('sigmoid', 0.5, 0.5),
    'tanh': ('tanh', 0.05, 0.5),
}

n_epochs = 1000
epsilon = 0.0

# ---------- FIGURA 1: y_real vs y_pred ----------
fig1, axes = plt.subplots(1, len(best_params), figsize=(18, 5))
axes = axes.flatten()

# ---------- FIGURA 2: evolución del error ----------
fig2, ax2 = plt.subplots(figsize=(10, 6))

for i, (combination_name, (activation_function, lr, beta)) in enumerate(best_params.items()):
    perceptron = Perceptron(n_inputs=X.shape[1], X_min=X_min, X_max=X_max, y_min=y_min, y_max=y_max, learning_rate=lr, activation_function=activation_function, beta=beta)
    error_history, conv_epoch = perceptron.train(X, y, epochs=n_epochs, epsilon=epsilon)

    y_pred = [perceptron.predict(xi)[0] for xi in X]

    # --- Figura 1: scatter y vs y_pred ---
    axes[i].scatter(y, y_pred, c="blue", alpha=0.6, label="Predicciones")
    axes[i].plot([y_min, y_max], [y_min, y_max], "r--", label="y = x (perfecto)")
    axes[i].set_xlabel("y real")
    axes[i].set_ylabel("y predicho")
    axes[i].set_title(f"{combination_name}")
    axes[i].legend()

    # --- Figura 2: curvas de error ---
    ax2.plot(error_history, label=f"{activation_function} (LR={lr}, Beta={beta})")

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
# -------------------------------------------------------------------------
# CAPACIDAD DE GENERALIZACIÓN: elegimos el mejor perceptron --> hacemos un simple train test split
# Sigmoid, LR=0.5, Beta=0.5
# -------------------------------------------------------------------------

# Train Perceptron SOLO con train
activation_function = 'sigmoid' 
lr = 0.5
beta = 0.5
n_epochs = 1000
epsilon = 0.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

perceptron = Perceptron(n_inputs=X.shape[1], X_min=X_min, X_max=X_max, y_min=y_min, y_max=y_max, learning_rate=lr, activation_function=activation_function, beta=beta)
error_history, conv_epoch = perceptron.train(X_train, y_train, epochs=n_epochs, epsilon=epsilon)

# --- Predicciones en TRAIN ---
y_train_pred = [perceptron.predict(xi)[0] for xi in X_train]
# --- Gráfico ---
plt.figure(figsize=(6,6))
plt.scatter(y_train, y_train_pred, c="blue", alpha=0.6, label="Predicciones (train)")
plt.plot([y_min, y_max], [y_min, y_max], "r--", label="y = x (perfecto)")
plt.xlabel("y real")
plt.ylabel("y predicho")
plt.title("Generalización del perceptrón (TRAIN)")
plt.legend()
plt.show()

# --- Predicciones en TEST ---
y_test_pred  = [perceptron.predict(xi)[0] for xi in X_test]
# --- Métricas ---
mse_train = np.mean((y_train - y_train_pred) ** 2)/len(y_train)
mse_test = np.mean((y_test - y_test_pred) ** 2)/len(y_test)
print(f"MSE Train: {mse_train:.4f}")
print(f"MSE Test: {mse_test:.4f}")

# --- Gráfico ---
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, c="blue", alpha=0.6, label="Predicciones (test)")
plt.plot([y_min, y_max], [y_min, y_max], "r--", label="y = x (perfecto)")
plt.xlabel("y real")
plt.ylabel("y predicho")
plt.title("Generalización del perceptrón (TEST)")
plt.legend()
plt.show()
"""


"""
# -------------------------------------------------------------------------
# CAPACIDAD DE GENERALIZACIÓN: elegimos el mejor perceptron --> hacemos k fold cross-validation
# Sigmoid, LR=0.5, Beta=0.5
# -------------------------------------------------------------------------

def cross_validate_perceptron(X, y, k, activation_function, lr, beta, epochs, epsilon):
    folds = k_fold_split(X, k)
    mse_train_list = []
    mse_test_list = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        perceptron = Perceptron(n_inputs=X.shape[1], X_min=X_min, X_max=X_max, y_min=y_min, y_max=y_max, learning_rate=lr, activation_function=activation_function, beta=beta)
        perceptron.train(X_train, y_train, epochs=n_epochs, epsilon=epsilon)

        # Predicciones
        y_train_pred = [perceptron.predict(xi)[0] for xi in X_train]
        y_test_pred  = [perceptron.predict(xi)[0] for xi in X_test]

        # Calcular MSE
        mse_train = np.mean((y_train - y_train_pred)**2)/len(y_train)
        mse_test  = np.mean((y_test - y_test_pred)**2)/len(y_test)

        print("len train set:", len(y_train))
        print("len test set:", len(y_test))

        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)

        print(f"Fold {fold+1}: MSE Train = {mse_train:.4f}, MSE Test = {mse_test:.4f}")

        # --- Gráfico ---
        plt.figure(figsize=(6,6))
        plt.scatter(y_train, y_train_pred, c="blue", alpha=0.6, label="Predicciones (train)")
        plt.scatter(y_test, y_test_pred, c="green", alpha=0.6, label="Predicciones (test)")
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


activation_function = 'sigmoid' 
lr = 0.5
beta = 0.5
n_epochs = 1000
epsilon = 0.0

k = 5

mse_train_cv, mse_test_cv = cross_validate_perceptron(X, y, k=k, activation_function=activation_function, lr=lr, beta=beta, epochs=n_epochs, epsilon=epsilon)

"""