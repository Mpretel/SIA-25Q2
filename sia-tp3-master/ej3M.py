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
        a = 1.0 / (1.0 + np.exp(-2.0 * z))   # sigmoide "ganancia 2"
        d = 2.0 * a * (1.0 - a)
    elif method == 'tanh':
        a = np.tanh(z)
        d = 1.0 - a**2
    elif method == 'relu':
        a = np.maximum(0.0, z)
        d = (z > 0.0).astype(z.dtype)
    else:
        raise ValueError(f"Unknown activation: {method}")
    return a, d

# =========================
# Inicialización
# =========================
def init_weights_with_bias(n_in, n_out, act, rng):
    """
    Devuelve W de tamaño (n_out, n_in+1) -> última columna es el bias.
    Xavier por defecto; He si la capa usa ReLU.
    """
    if act == 'relu':
        limit = np.sqrt(6.0 / n_in)  # He uniform (aprox.)
    else:
        limit = np.sqrt(6.0 / (n_in + n_out))  # Xavier/Glorot uniform
    W = rng.uniform(-limit, limit, size=(n_out, n_in + 1))  # +1 por bias
    return W

# =========================
# MLP vectorizado con bias absorbido + mini-batch + optimizadores
# =========================
class MLPMatrixBias:
    def __init__(
        self,
        layer_sizes,
        activations,
        learning_rate=0.1,
        seed=0,
        optimizer='sgd',
        momentum_beta=0.9,            # para Momentum
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8  # para Adam
    ):
        """
        layer_sizes: [n_in, h1, ..., n_out] (SIN +1; el +1 lo agregamos internamente por bias)
        activations: ['tanh', ..., 'sigmoid']  # una por capa
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
        self._adam_t = 0  # step global para bias-correction

        rng = np.random.default_rng(seed)
        self.W = []  # cada W[l] tiene forma (n_out, n_in+1)  (última col = bias)
        for l in range(len(activations)):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l+1]
            Wl = init_weights_with_bias(n_in, n_out, activations[l], rng)
            self.W.append(Wl)

        # Estados para optimizadores
        if self.optimizer == 'momentum':
            # v (velocidad) por capa (misma forma que W)
            self.V = [np.zeros_like(Wl) for Wl in self.W]
        elif self.optimizer == 'adam':
            # m y v (1er y 2do momento) por capa
            self.M = [np.zeros_like(Wl) for Wl in self.W]
            self.V = [np.zeros_like(Wl) for Wl in self.W]

    # ---------- Forward (una muestra) ----------
    def forward(self, x):
        """
        x: vector (n_in,)
        Devuelve:
          A: [A0, A1, ..., AL]           (activaciones sin el 1; A0=x)
          Z: [Z1, ..., ZL]
          D: [D1, ..., DL]
          A_aug: [A0_aug, ..., A(L-1)_aug] (activaciones con 1 añadido para cada capa)
        """
        a = np.asarray(x, dtype=float).reshape(-1)
        A = [a]
        A_aug = []
        Z, D = [], []

        for l, Wl in enumerate(self.W):
            a_aug = np.append(a, 1.0)             # (n_in+1,)
            z = Wl @ a_aug                        # (n_out,)
            a, d = act_forward(z, self.activations[l])

            A_aug.append(a_aug)
            Z.append(z)
            D.append(d)
            A.append(a)

        return A, Z, D, A_aug

    # ---------- Gradientes (una muestra) ----------
    def sample_gradients(self, A, D, A_aug, y_true):
        """
        Devuelve lista dW por capa (misma forma que W), para una sola muestra.
        """
        L = len(self.W)
        deltas = [None] * L

        # Delta en salida
        aL = A[-1]
        y = np.asarray(y_true, dtype=float).reshape(-1)
        if self.activations[-1] == 'sigmoid' and aL.size == 1:
            deltas[-1] = (aL - y)               # BCE simplificada con sigmoide 1D
        else:
            deltas[-1] = (aL - y) * D[-1]       # MSE genérico

        # Deltas ocultas
        for l in range(L-2, -1, -1):
            back = self.W[l+1].T @ deltas[l+1]  # (n_in+1,)
            deltas[l] = back[:-1] * D[l]        # descarto bias, * derivada

        # dW por capa
        dW = [np.outer(deltas[l], A_aug[l]) for l in range(L)]
        return dW

    # ---------- Aplicar gradiente con optimizador ----------
    def apply_gradients(self, dW):
        """
        dW: lista (por capa) de gradientes (mismas formas que W)
        """
        lr = self.learning_rate

        if self.optimizer == 'sgd':
            for l in range(len(self.W)):
                self.W[l] -= lr * dW[l]

        elif self.optimizer == 'momentum':
            beta = self.momentum_beta
            for l in range(len(self.W)):
                self.V[l] = beta * self.V[l] + (1.0 - 0.0) * dW[l]  # clásico: v = beta*v + grad
                self.W[l] -= lr * self.V[l]

        elif self.optimizer == 'adam':
            b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_eps
            self._adam_t += 1
            for l in range(len(self.W)):
                # momentos
                self.M[l] = b1 * self.M[l] + (1.0 - b1) * dW[l]
                self.V[l] = b2 * self.V[l] + (1.0 - b2) * (dW[l] * dW[l])
                # bias correction
                m_hat = self.M[l] / (1.0 - (b1 ** self._adam_t))
                v_hat = self.V[l] / (1.0 - (b2 ** self._adam_t))
                # update
                self.W[l] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ---------- Pérdida ----------
    def loss(self, y_pred, y_true):
        y = np.asarray(y_true, dtype=float).reshape(-1)
        if self.activations[-1] == 'sigmoid' and y_pred.size == 1:
            eps = 1e-12
            a = np.clip(y_pred[0], eps, 1.0 - eps)
            yt = y[0]
            return -(yt*np.log(a) + (1.0-yt)*np.log(1.0-a))  # BCE
        else:
            return 0.5 * np.sum((y_pred - y)**2)             # MSE

    # ---------- Fit ----------
    def fit(self, X, Y, epochs=1000, epsilon=1e-3, verbose=True, shuffle=True, batch_size=1):
        """
        batch_size=1  -> SGD (online)
        batch_size=N  -> mini-batch (N)
        batch_size=len(X) -> batch completo
        """
        X = [np.asarray(x, dtype=float).reshape(-1) for x in X]
        Y = [np.asarray(y, dtype=float).reshape(-1) for y in Y]

        n = len(X)
        idx = list(range(n))

        for epoch in range(1, epochs+1):
            if shuffle:
                random.shuffle(idx)

            total_loss = 0.0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = idx[start:end]

                # acumular gradientes del batch
                dW_sum = [np.zeros_like(Wl) for Wl in self.W]

                for t in batch_idx:
                    A, Z, D, A_aug = self.forward(X[t])
                    total_loss += float(self.loss(A[-1], Y[t]))
                    dW = self.sample_gradients(A, D, A_aug, Y[t])
                    for l in range(len(self.W)):
                        dW_sum[l] += dW[l]

                # promedio del batch y paso del optimizador
                m = float(len(batch_idx))
                dW_avg = [g / m for g in dW_sum]
                self.apply_gradients(dW_avg)

            if verbose and (epoch == 1 or epoch % 50 == 0):
                print(f"Epoch {epoch} - Loss: {total_loss:.6f}")

            if total_loss <= epsilon:
                if verbose:
                    print(f"Convergió en epoch {epoch} con loss={total_loss:.6f}")
                break

    # ---------- Predict ----------
    def predict(self, x):
        A, _, _, _ = self.forward(x)
        return A[-1]


# =========================
# Demo: XOR (2 -> 3 -> 1)
# =========================
if __name__ == "__main__":
    X = [[-1, -1], [-1,  1], [ 1, -1], [ 1,  1]]
    Y = [[0], [1], [1], [0]]

    # Probá cambiando optimizer a 'sgd' | 'momentum' | 'adam'
    mlp = MLPMatrixBias(
        layer_sizes=[2, 3, 1],
        activations=['tanh', 'sigmoid'],
        learning_rate=0.2,
        seed=0,
        optimizer='adam',          # <--- elegí aquí
        momentum_beta=0.9,         # usado si optimizer='momentum'
        adam_beta1=0.9,            # usados si optimizer='adam'
        adam_beta2=0.999,
        adam_eps=1e-8
    )

    mlp.fit(X, Y, epochs=5000, epsilon=1e-3, verbose=True, batch_size=4)

    print("\nPredicciones XOR:")
    for x in X:
        yhat = mlp.predict(x)[0]
        print(x, "->", round(float(yhat), 4), "(clase:", int(yhat >= 0.5), ")")
