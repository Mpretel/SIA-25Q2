import numpy as np
import random

# =========================
# Activaciones (vectorizadas)
# =========================
def act_forward(z, method):
    if method == 'linear':
        a = z
        d = np.ones_like(z)
    elif method == 'sigmoid':
        # sigmoide con "ganancia 2" (como venías usando)
        a = 1.0 / (1.0 + np.exp(-2.0 * z))
        d = 2.0 * a * (1.0 - a)
    elif method == 'tanh':
        a = np.tanh(z)
        d = 1.0 - a**2
    elif method == 'relu':
        a = np.maximum(0.0, z)
        d = (z > 0.0).astype(z.dtype)
    elif method == 'step':
        a = np.where(z >= 0.0, 1.0, -1.0)   # NO usar para backprop en práctica
        d = np.zeros_like(z)
    else:
        raise ValueError(f"Unknown activation: {method}")
    return a, d

# =========================
# Inicializaciones
# =========================
def init_weights(n_in, n_out, act, rng):
    """
    Xavier por defecto; He si la capa usa ReLU.
    Devuelve W[n_out, n_in], b[n_out]
    """
    if act == 'relu':
        # He uniform
        limit = np.sqrt(6.0 / n_in)
    else:
        # Xavier/Glorot uniform
        limit = np.sqrt(6.0 / (n_in + n_out))
    W = rng.uniform(-limit, limit, size=(n_out, n_in))
    b = np.zeros((n_out,), dtype=float)
    return W, b

# =========================
# MLP vectorizado
# =========================
class MLPMatrix:
    def __init__(self, layer_sizes, activations, learning_rate=0.1, seed=0):
        """
        layer_sizes: [n_in, h1, h2, ..., n_out]
        activations: ['tanh', ..., 'sigmoid']  # una por capa (sin contar entrada)
        """
        assert len(layer_sizes) >= 2
        assert len(activations) == len(layer_sizes) - 1

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = float(learning_rate)

        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for l in range(len(activations)):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l+1]
            Wl, bl = init_weights(n_in, n_out, activations[l], rng)
            self.W.append(Wl)
            self.b.append(bl)

    # ---------- Forward ----------
    def forward(self, x):
        """
        x: vector (n_in,)
        Devuelve:
          A: [A0=x, A1, ..., AL]    (cada A es vector)
          Z: [Z1, ..., ZL]          (cada Z es vector)
          D: [D1, ..., DL]          (cada D es vector con derivadas)
        """
        a = np.asarray(x, dtype=float).reshape(-1)
        A = [a]
        Z, D = [], []
        for l, (Wl, bl) in enumerate(zip(self.W, self.b)):
            z = Wl @ a + bl
            a, d = act_forward(z, self.activations[l])
            Z.append(z)
            D.append(d)
            A.append(a)
        return A, Z, D

    # ---------- Backprop + Update ----------
    def backward_update(self, A, D, y_true):
        """
        A, D: listas del forward
        y_true: vector (n_out,)
        Reglas (vectorizadas):
          W[l] -= lr * outer(delta[l], A[l])
          b[l] -= lr * delta[l]
        """
        L = len(self.W)
        deltas = [None] * L

        # Delta salida
        aL = A[-1]
        y = np.asarray(y_true, dtype=float).reshape(-1)
        if self.activations[-1] == 'sigmoid':
            # BCE simplifica: delta = a - y
            deltas[-1] = (aL - y)
        else:
            deltas[-1] = (aL - y) * D[-1]

        # Deltas ocultas
        for l in range(L-2, -1, -1):
            deltas[l] = (self.W[l+1].T @ deltas[l+1]) * D[l]

        # Actualización matricial
        for l in range(L):
            self.W[l] -= self.learning_rate * np.outer(deltas[l], A[l])
            self.b[l] -= self.learning_rate * deltas[l]

    # ---------- Pérdida ----------
    def loss(self, y_pred, y_true):
        y = np.asarray(y_true, dtype=float).reshape(-1)
        if self.activations[-1] == 'sigmoid' and y_pred.size == 1:
            # BCE binaria (1 salida)
            eps = 1e-12
            a = np.clip(y_pred[0], eps, 1.0 - eps)
            yt = y[0]
            return -(yt*np.log(a) + (1.0-yt)*np.log(1.0-a))
        else:
            # MSE
            return 0.5 * np.sum((y_pred - y)**2)

    # ---------- Fit ----------
    def fit(self, X, Y, epochs=1000, epsilon=1e-3, verbose=True, shuffle=True):
        X = [np.asarray(x, dtype=float).reshape(-1) for x in X]
        Y = [np.asarray(y, dtype=float).reshape(-1) for y in Y]

        n = len(X)
        idx = list(range(n))
        for epoch in range(1, epochs+1):
            if shuffle:
                random.shuffle(idx)

            total_loss = 0.0
            for t in idx:
                A, Z, D = self.forward(X[t])
                total_loss += float(self.loss(A[-1], Y[t]))
                self.backward_update(A, D, Y[t])

            if verbose and (epoch == 1 or epoch % 50 == 0):
                print(f"Epoch {epoch} - Loss: {total_loss:.6f}")

            if total_loss <= epsilon:
                if verbose:
                    print(f"Convergió en epoch {epoch} con loss={total_loss:.6f}")
                break

    # ---------- Predict ----------
    def predict(self, x):
        A, _, _ = self.forward(x)
        return A[-1]  # vector salida


# =========================
# Demo: XOR (2 -> 3 -> 1)
# =========================
if __name__ == "__main__":
    X = [[-1, -1], [-1,  1], [ 1, -1], [ 1,  1]]
    Y = [[0], [1], [1], [0]]

    mlp = MLPMatrix(
        layer_sizes=[2, 3, 1],
        activations=['tanh', 'sigmoid'],  # ocultas tanh, salida sigmoid
        learning_rate=0.2,
        seed=0
    )

    mlp.fit(X, Y, epochs=5000, epsilon=1e-3, verbose=True)

    print("\nPredicciones XOR:")
    for x in X:
        yhat = mlp.predict(x)[0]
        print(x, "->", round(float(yhat), 4), "(clase:", int(yhat >= 0.5), ")")
