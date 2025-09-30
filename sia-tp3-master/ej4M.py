import os, gzip, struct, urllib.request
import numpy as np
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=INFO off, 2=WARNING off, 3=ERROR only
# opcional: si querés quitar el aviso de oneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from keras.datasets import mnist  # recién ahora importás keras/tf

# # =========================
# # Utilidades: MNIST (IDX)
# # =========================
# MNIST_URLS = {
#     "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
#     "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
#     "test_images":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
#     "test_labels":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
# }


# def download_if_needed(path, url):
#     if not os.path.exists(path):
#         print(f"Descargando {url} ...")
#         urllib.request.urlretrieve(url, path)
#     else:
#         pass

# def load_idx_images(gz_path):
#     with gzip.open(gz_path, 'rb') as f:
#         magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
#         assert magic == 2051
#         buf = f.read(rows * cols * num)
#         data = np.frombuffer(buf, dtype=np.uint8)
#         data = data.reshape(num, rows * cols).astype(np.float32) / 255.0
#         return data  # (N, 784)

# def load_idx_labels(gz_path):
#     with gzip.open(gz_path, 'rb') as f:
#         magic, num = struct.unpack(">II", f.read(8))
#         assert magic == 2041 or magic == 2049  # (algunas descargas cambian reportes)
#         buf = f.read(num)
#         labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#         return labels  # (N,)

# def one_hot(y, n_classes):
#     Y = np.zeros((y.shape[0], n_classes), dtype=np.float32)
#     Y[np.arange(y.shape[0]), y] = 1.0
#     return Y

# =========================
# Activaciones (vectorizadas)
# =========================
def act_forward(z, method):
    if method == 'linear':
        a = z
        d = np.ones_like(z)
    elif method == 'sigmoid':
        a = 1.0 / (1.0 + np.exp(-2.0 * z))
        d = 2.0 * a * (1.0 - a)
    elif method == 'tanh':
        a = np.tanh(z)
        d = 1.0 - a**2
    elif method == 'relu':
        a = np.maximum(0.0, z)
        d = (z > 0.0).astype(z.dtype)
    elif method == 'softmax':
        # estabilidad numérica
        z_shift = z - np.max(z)
        exp_z = np.exp(z_shift)
        a = exp_z / np.sum(exp_z)
        d = None  # no se usa; con CE, delta = a - y
    else:
        raise ValueError(f"Unknown activation: {method}")
    return a, d

# =========================
# Inicialización
# =========================
def init_weights_with_bias(n_in, n_out, act, rng):
    """
    W de tamaño (n_out, n_in+1) -> última columna es el bias.
    He para ReLU, Xavier para otras.
    """
    if act == 'relu':
        limit = np.sqrt(6.0 / n_in)
    else:
        limit = np.sqrt(6.0 / (n_in + n_out))
    W = rng.uniform(-limit, limit, size=(n_out, n_in + 1))
    return W

# =========================
# MLP vectorizado + bias absorbido + mini-batch + optimizadores + softmax
# =========================
class MLPMatrixBias:
    def __init__(
        self,
        layer_sizes,
        activations,
        learning_rate=1e-3,
        seed=0,
        optimizer='adam',
        momentum_beta=0.9,
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        l2_lambda=0.0  # regularización L2 (no penaliza bias)
    ):
        """
        layer_sizes: [n_in, h1, ..., n_out]
        activations: ['relu', ..., 'softmax']  # una por capa (salida softmax p/ multiclase)
        optimizer: 'sgd' | 'momentum' | 'adam'
        """
        assert len(layer_sizes) >= 2
        assert len(activations) == len(layer_sizes) - 1
        assert optimizer in ('sgd', 'momentum', 'adam')

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = float(learning_rate)

        self.optimizer = optimizer
        self.momentum_beta = momentum_beta
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self._adam_t = 0
        self.l2_lambda = float(l2_lambda)

        rng = np.random.default_rng(seed)
        self.W = [init_weights_with_bias(layer_sizes[l], layer_sizes[l+1], activations[l], rng)
                  for l in range(len(activations))]

        # estados optimizadores
        if self.optimizer == 'momentum':
            self.V = [np.zeros_like(Wl) for Wl in self.W]
        elif self.optimizer == 'adam':
            self.M = [np.zeros_like(Wl) for Wl in self.W]
            self.V = [np.zeros_like(Wl) for Wl in self.W]

    # ---------- Forward (una muestra) ----------
    def forward(self, x):
        a = np.asarray(x, dtype=np.float32).reshape(-1)
        A, A_aug, Z, D = [a], [], [], []

        for l, Wl in enumerate(self.W):
            a_aug = np.append(a, 1.0)      # agrega bias
            z = Wl @ a_aug
            a, d = act_forward(z, self.activations[l])
            A_aug.append(a_aug)
            Z.append(z)
            D.append(d)
            A.append(a)

        return A, Z, D, A_aug

    # ---------- Gradientes (una muestra) ----------
    def sample_gradients(self, A, D, A_aug, y_true):
        """
        dW por capa; incluye L2 sobre pesos (excepto última col de bias).
        """
        L = len(self.W)
        deltas = [None]*L

        aL = A[-1]
        y = np.asarray(y_true, dtype=np.float32).reshape(-1)

        # salida: softmax + cross-entropy => delta = a - y
        if self.activations[-1] == 'softmax':
            deltas[-1] = (aL - y)
        else:
            # otros casos: MSE
            deltas[-1] = (aL - y) * D[-1]

        # ocultas
        for l in range(L-2, -1, -1):
            back = self.W[l+1].T @ deltas[l+1]     # (n_in+1,)
            deltas[l] = back[:-1] * D[l]           # descarta bias

        # gradientes
        dW = [np.outer(deltas[l], A_aug[l]) for l in range(L)]

        # L2 (sin penalizar bias: última columna a 0 en el término de reg)
        if self.l2_lambda > 0.0:
            for l in range(L):
                reg = self.W[l].copy()
                reg[:, -1] = 0.0                   # no penalizar bias
                dW[l] += self.l2_lambda * reg
        return dW

    # ---------- Aplicar gradiente ----------
    def apply_gradients(self, dW):
        lr = self.learning_rate
        if self.optimizer == 'sgd':
            for l in range(len(self.W)):
                self.W[l] -= lr * dW[l]
        elif self.optimizer == 'momentum':
            beta = self.momentum_beta
            for l in range(len(self.W)):
                self.V[l] = beta * self.V[l] + dW[l]
                self.W[l] -= lr * self.V[l]
        elif self.optimizer == 'adam':
            b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_eps
            self._adam_t += 1
            for l in range(len(self.W)):
                self.M[l] = b1 * self.M[l] + (1.0 - b1) * dW[l]
                self.V[l] = b2 * self.V[l] + (1.0 - b2) * (dW[l] * dW[l])
                m_hat = self.M[l] / (1.0 - (b1 ** self._adam_t))
                v_hat = self.V[l] / (1.0 - (b2 ** self._adam_t))
                self.W[l] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ---------- Pérdida ----------
    def loss(self, y_pred, y_true):
        y = np.asarray(y_true, dtype=np.float32).reshape(-1)
        if self.activations[-1] == 'softmax':
            # cross-entropy multiclase
            eps = 1e-12
            a = np.clip(y_pred, eps, 1.0 - eps)
            return -np.sum(y * np.log(a))
        else:
            return 0.5 * np.sum((y_pred - y)**2)

    # ---------- Fit ----------
    def fit(self, X, Y, epochs=10, epsilon=1e-3, verbose=True, shuffle=True, batch_size=128,
            X_val=None, Y_val=None, early_patience=5):
        X = [np.asarray(x, dtype=np.float32).reshape(-1) for x in X]
        Y = [np.asarray(y, dtype=np.float32).reshape(-1) for y in Y]

        n = len(X)
        idx = list(range(n))

        best_val = np.inf
        patience = early_patience

        for epoch in range(1, epochs+1):
            if shuffle:
                random.shuffle(idx)

            total_loss = 0.0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = idx[start:end]

                # acumular grad del batch
                dW_sum = [np.zeros_like(Wl) for Wl in self.W]
                for t in batch_idx:
                    A, Z, D, A_aug = self.forward(X[t])
                    total_loss += float(self.loss(A[-1], Y[t]))
                    dW = self.sample_gradients(A, D, A_aug, Y[t])
                    for l in range(len(self.W)):
                        dW_sum[l] += dW[l]

                m = float(len(batch_idx))
                dW_avg = [g / m for g in dW_sum]
                self.apply_gradients(dW_avg)

            # validación/early stopping
            val_loss = None
            if X_val is not None and Y_val is not None:
                val_loss = self.evaluate_loss(X_val, Y_val, batch_size=512)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    patience = early_patience
                else:
                    patience -= 1

            if verbose:
                if val_loss is not None:
                    print(f"Epoch {epoch:3d} - TrainLoss: {total_loss:.4f} | ValLoss: {val_loss:.4f} | Patience: {patience}")
                else:
                    print(f"Epoch {epoch:3d} - TrainLoss: {total_loss:.4f}")

            if total_loss <= epsilon or (X_val is not None and patience <= 0):
                if verbose:
                    print("Stop por criterio de convergencia/early stopping.")
                break

    def evaluate_loss(self, X, Y, batch_size=512):
        X = [np.asarray(x, dtype=np.float32).reshape(-1) for x in X]
        Y = [np.asarray(y, dtype=np.float32).reshape(-1) for y in Y]
        total = 0.0
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            for t in range(start, end):
                A, _, _, _ = self.forward(X[t])
                total += float(self.loss(A[-1], Y[t]))
        return total

    # ---------- Predict ----------
    def predict_proba(self, x):
        A, _, _, _ = self.forward(x)
        return A[-1]

    def predict_class(self, x):
        return int(np.argmax(self.predict_proba(x)))

# =========================
# Entrenamiento MNIST
# =========================
# def main():
#     data_dir = "./mnist_data"
#     os.makedirs(data_dir, exist_ok=True)

#     # Descargas
#     paths = {}
#     for k, url in MNIST_URLS.items():
#         fn = os.path.join(data_dir, os.path.basename(url))
#         download_if_needed(fn, url)
#         paths[k] = fn

#     # Carga
#     X_train = load_idx_images(paths["train_images"])   # (60000, 784) float32 [0,1]
#     y_train = load_idx_labels(paths["train_labels"])   # (60000,)
#     X_test  = load_idx_images(paths["test_images"])    # (10000, 784)
#     y_test  = load_idx_labels(paths["test_labels"])    # (10000,)

#     # One-hot
#     Y_train = one_hot(y_train, 10)
#     Y_test  = one_hot(y_test, 10)

#     # Split simple train/val
#     val_frac = 0.1
#     n_train = int((1.0 - val_frac) * X_train.shape[0])
#     X_tr, Y_tr = X_train[:n_train], Y_train[:n_train]
#     X_val, Y_val = X_train[n_train:], Y_train[n_train:]

#     # Modelo
#     mlp = MLPMatrixBias(
#         layer_sizes=[784, 256, 10],
#         activations=['relu', 'softmax'],
#         learning_rate=1e-3,     # Adam típico
#         seed=0,
#         optimizer='adam',
#         l2_lambda=1e-4          # L2 leve (no penaliza bias)
#     )

#     # Entrenar
#     mlp.fit(
#         X_tr, Y_tr,
#         epochs=20,
#         batch_size=128,
#         X_val=X_val, Y_val=Y_val,
#         early_patience=5,
#         verbose=True
#     )

#     # Evaluación
#     test_acc = accuracy(mlp, X_test, y_test, batch_size=512)
#     print(f"\nTest accuracy: {test_acc*100:.2f}%")

#     # (Opcional) Matriz de confusión
#     cm = confusion_matrix(mlp, X_test, y_test, num_classes=10)
#     print("\nConfusion matrix:\n", cm)

# def accuracy(model, X, y, batch_size=1024):
#     correct = 0
#     n = len(X)
#     for i in range(n):
#         pred = model.predict_class(X[i])
#         if pred == int(y[i]):
#             correct += 1
#     return correct / n

# def confusion_matrix(model, X, y, num_classes=10):
#     cm = np.zeros((num_classes, num_classes), dtype=int)
#     for i in range(len(X)):
#         pred = model.predict_class(X[i])
#         cm[int(y[i]), pred] += 1
#     return cm

# if __name__ == "__main__":
#     main()



# =========================

# =========================
# Main con Keras MNIST
# =========================
def one_hot(y, n_classes=10):
    Y = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y

def accuracy(model, X, y):
    ok = 0
    for i in range(len(X)):
        pred = np.argmax(model.predict_proba(X[i]))
        ok += int(pred == int(y[i]))
    return ok / len(X)

def confusion_matrix(model, X, y, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(X)):
        pred = np.argmax(model.predict_proba(X[i]))
        cm[int(y[i]), pred] += 1
    return cm

def main():
    # 1) Cargar MNIST con Keras (descarga/caché automática en ~/.keras/datasets)
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
    print(f"Train: {x_train.shape}, {y_train.shape} | Test: {x_test.shape}, {y_test.shape}")
    # 2) Normalizar y aplanar a vectores de 784
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test  = x_test.reshape((x_test.shape[0],  -1))

    # 3) One-hot para softmax
    Y_train = one_hot(y_train, 10)
    Y_test  = one_hot(y_test, 10)

    # 4) Split train/val
    val_frac = 0.1
    n_train = int((1.0 - val_frac) * x_train.shape[0])
    X_tr, Y_tr = x_train[:n_train], Y_train[:n_train]
    X_val, Y_val = x_train[n_train:], Y_train[n_train:]

    # 5) Instanciar MLP (relu + softmax) con Adam
    mlp = MLPMatrixBias(
        layer_sizes=[784, 10, 10],
        activations=['relu', 'softmax'],
        learning_rate=1e-3,
        optimizer='adam',          # 'sgd' | 'momentum' | 'adam'
        seed=0,
        l2_lambda=1e-4
    )

    # 6) Entrenar
    mlp.fit(
        X_tr, Y_tr,
        epochs=3,
        batch_size=128,
        X_val=X_val, Y_val=Y_val,
        early_patience=5,
        verbose=True
    )

    # 7) Evaluar
    test_acc = accuracy(mlp, x_test, y_test)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    cm = confusion_matrix(mlp, x_test, y_test, num_classes=10)
    print("\nConfusion matrix:\n", cm)

if __name__ == "__main__":
    main()
# =========================


