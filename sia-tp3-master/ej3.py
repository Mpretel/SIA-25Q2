import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



SEED = 42
np.random.seed(SEED)


# ------------------------------
# Funciones de activación
# ------------------------------
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

activation_functions = {
    'tanh': (tanh, tanh_derivative, (-1, 1), 0.0),
    'sigmoid': (sigmoid, sigmoid_derivative, (0, 1), 0.5)
}


# ------------------------------
# Clase MLP
# ------------------------------
class MLP:
    def __init__(self, n_input, n_hidden, n_output, activation_function='tanh', learning_rate=0.1, optimizer='gd', beta=0.9, beta1=0.9, beta2=0.999, epsilon_adam=1e-8):
        # Pesos iniciales
        self.W_hidden = np.random.uniform(-0.1, 0.1, (n_input, n_hidden))
        self.W_output = np.random.uniform(-0.1, 0.1, (n_hidden, n_output))
        self.learning_rate = learning_rate
        self.activation_function_name = activation_function
        self.activation_func, self.activation_deriv, self.range, self.threshold = activation_functions[activation_function]

        # Optimizer
        self.optimizer = optimizer
        self.beta = beta            # para momentum
        self.beta1 = beta1          # Adam
        self.beta2 = beta2          # Adam
        self.epsilon_adam = epsilon_adam
        self.iteration = 0          # para Adam

        # Inicializar momentos
        self.vW_hidden = np.zeros_like(self.W_hidden)
        self.vW_output = np.zeros_like(self.W_output)
        self.mW_hidden = np.zeros_like(self.W_hidden)  # para Adam
        self.vW_hidden_adam = np.zeros_like(self.W_hidden)
        self.mW_output = np.zeros_like(self.W_output)
        self.vW_output_adam = np.zeros_like(self.W_output)

    def forward(self, X):
        """Forward pass"""
        self.hidden_input = np.dot(X, self.W_hidden)
        self.hidden_output = self.activation_func(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.W_output)
        self.final_output = self.activation_func(self.final_input)
        return self.final_output


    def backward(self, X, y, output):
        """Backpropagation & weight update using specified optimizer"""
        error = y - output
        d_output = error * self.activation_deriv(self.final_input)
        d_hidden = d_output.dot(self.W_output.T) * self.activation_deriv(self.hidden_input)

        # Gradientes
        grad_W_output = self.hidden_output.T.dot(d_output)
        grad_W_hidden = X.T.dot(d_hidden)

        # Weight update
        if self.optimizer == 'gd': # Gradient descent
            self.W_output += self.learning_rate * grad_W_output
            self.W_hidden += self.learning_rate * grad_W_hidden

        elif self.optimizer == 'momentum':
            self.vW_output = self.beta * self.vW_output + (1 - self.beta) * grad_W_output
            self.vW_hidden = self.beta * self.vW_hidden + (1 - self.beta) * grad_W_hidden
            self.W_output += self.learning_rate * self.vW_output
            self.W_hidden += self.learning_rate * self.vW_hidden

        elif self.optimizer == 'adam':
            self.iteration += 1
            # Actualizar momentums y variances
            self.mW_output = self.beta1 * self.mW_output + (1 - self.beta1) * grad_W_output
            self.vW_output_adam = self.beta2 * self.vW_output_adam + (1 - self.beta2) * (grad_W_output**2)
            self.mW_hidden = self.beta1 * self.mW_hidden + (1 - self.beta1) * grad_W_hidden
            self.vW_hidden_adam = self.beta2 * self.vW_hidden_adam + (1 - self.beta2) * (grad_W_hidden**2)
            # Corrección de sesgo
            m_hat_out = self.mW_output / (1 - self.beta1**self.iteration)
            v_hat_out = self.vW_output_adam / (1 - self.beta2**self.iteration)
            m_hat_hid = self.mW_hidden / (1 - self.beta1**self.iteration)
            v_hat_hid = self.vW_hidden_adam / (1 - self.beta2**self.iteration)
            # Actualización
            self.W_output += self.learning_rate * m_hat_out / (np.sqrt(v_hat_out) + self.epsilon_adam)
            self.W_hidden += self.learning_rate * m_hat_hid / (np.sqrt(v_hat_hid) + self.epsilon_adam)

        return np.sum(error**2)

    def train(self, X, y, epochs=1000, epsilon=0.0, batch_size=None):
        """MLP training: 
        - if batch_size is None, use full batch training
        - if batch_size=1, use online training
        - else use mini-batch training with a size of batch_size
        """
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples  # batch

        loss_history = []

        for epoch in range(epochs):
            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[idx], y[idx]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                Xi, yi = X_shuffled[start:end], y_shuffled[start:end]

                # Forward + Backward
                output = self.forward(Xi)
                loss = self.backward(Xi, yi, output)

            loss_history.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss={loss:.4f}")

            if loss <= epsilon:
                print(f"Early stopping at epoch {epoch}, Loss={loss:.4f}")
                return loss_history

        return loss_history

    def predict(self, X):
        """Predicción binaria"""
        output = self.forward(X)

        if self.range == (-1, 1):
            predictions = np.where(output >= self.threshold, 1, -1)
        elif self.range == (0, 1):
            predictions = np.where(output >= self.threshold, 1, 0)

        return predictions


"""
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
mlp = MLP(n_input=3, n_hidden=3, n_output=1, learning_rate=0.1, activation_function='tanh', optimizer='momentum')

# Entrenar
loss_history = mlp.train(X, y, epochs=1000, epsilon=0.0, batch_size=4)

# Predicciones finales
final_output = mlp.forward(X)
predictions = mlp.predict(X)

print(f"Predicciones continuas ({mlp.activation_function_name}):")
print(final_output)
print("Predicciones finales (-1/1):")
print(predictions)
print("Salida esperada:")
print(y)

# plot loss history
plt.plot(loss_history)
plt.xlabel("Epoch") 
plt.ylabel("Loss")
plt.title("Curva de error durante el entrenamiento")
plt.show()





# Lista de optimizadores
optimizers = ['gd', 'momentum', 'adam']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8,5))

for opt, c in zip(optimizers, colors):
    # Crear MLP con el optimizador actual
    mlp = MLP(n_input=3, n_hidden=3, n_output=1,
              learning_rate=0.1,
              activation_function='tanh',
              optimizer=opt)
    
    # Entrenar
    loss_history = mlp.train(X, y, epochs=1000, epsilon=0.0, batch_size=4)
    
    # Graficar curva de error
    plt.plot(loss_history, color=c, label=opt.upper())

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Comparación de evolución del error (GD vs Momentum vs Adam)")
plt.legend()
plt.grid(True)
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


# ------------------------------
# Discriminar paridad
# ------------------------------

# Etiquetas: pares = 1, impares = 0
y = np.array([[1 if d % 2 == 0 else 0] for d in range(n_digits)])

# Crear y entrenar MLP
mlp = MLP(n_input=35, n_hidden=10, n_output=1, learning_rate=0.1, activation_function='sigmoid', optimizer='adam')
loss_history = mlp.train(X, y, epochs=1000, epsilon=0.0, batch_size=10)

pred = mlp.predict(X)
print("Esperado:", y.flatten())
print("Predicho:", pred.flatten())

# Plot loss history
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de error durante el entrenamiento")
plt.show()


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


# Crear y entrenar MLP
mlp = MLP(n_input=35, n_hidden=10, n_output=10, learning_rate=0.1, activation_function='sigmoid', optimizer='momentum')
loss_history = mlp.train(X, y, epochs=1000, epsilon=0.0, batch_size=1)

y_pred = mlp.predict(X)
y_pred_label = np.argmax(y_pred, axis=1)


print("Esperado:", y_label)
print("Predicho:", y_pred_label)

# Plot loss history
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de error durante el entrenamiento")
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
    y_pred_noisy_label = np.argmax(mlp.predict(X_noisy), axis=1)
    
    for j in range(10):
        axes[i, j].imshow(
            X_noisy[j].reshape(digit_height, digit_width),
            cmap='gray_r',  # inverso: 0=blanco, 1=negro
            vmin=0, vmax=1
        )
        axes[i, j].axis('off')

plt.suptitle("Imágenes de los números con distintos niveles de ruido")
plt.show()




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



for nl in noise_levels:
    X_noisy = add_noise(X, noise_level=nl)
    y_pred_noisy = mlp.predict(X_noisy)
    y_pred_noisy_label = np.argmax(y_pred_noisy, axis=1)
    
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
        y_pred_noisy = mlp.predict(X_noisy)
        y_pred_noisy_label = np.argmax(y_pred_noisy, axis=1)

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
    
    sns.heatmap(cm_rel, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title(f"Relativa - Ruido {nl*100:.1f}%")
    axes[1].set_xlabel("Predicho")
    axes[1].set_ylabel("Esperado")
    
    plt.tight_layout()
    plt.show()
