import numpy as np

# ----- Datos XOR -----
X = np.array([[1, -1, 1],
              [1, 1, -1],
              [1, -1, -1],
              [1, 1, 1]])
y = np.array([[1],
              [1],
              [-1],
              [-1]])

# ----- Hiperparámetros -----
n_input = 3        # entradas
n_hidden = 3       # neuronas en capa oculta
n_output = 1       # salida
learning_rate = 0.1
n_epochs = 1000

# ----- Inicialización de pesos -----
np.random.seed(42)
W_hidden = np.random.uniform(-0.1, 0.1, (n_input, n_hidden))
W_output = np.random.uniform(-0.1, 0.1, (n_hidden, n_output))

# ----- Funciones de activación -----
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# ----- Entrenamiento -----
batch_size = 4
for epoch in range(n_epochs):
    # Mezclar los datos al inicio de cada época
    idx = np.random.permutation(len(X))
    X_shuffled = X[idx]
    y_shuffled = y[idx]

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        Xi = X_shuffled[start:end]
        yi = y_shuffled[start:end]

        # Forward
        hidden_output = tanh(np.dot(Xi, W_hidden))
        final_output = tanh(np.dot(hidden_output, W_output))

        # Error
        error = yi - final_output

        # Backprop
        d_output = error * tanh_derivative(final_output)
        d_hidden = d_output.dot(W_output.T) * tanh_derivative(hidden_output)

        # Actualización de pesos (mini-batch)
        W_output += hidden_output.T.dot(d_output) * learning_rate
        W_hidden += Xi.T.dot(d_hidden) * learning_rate

# ----- Resultados -----
hidden_input = np.dot(X, W_hidden)
hidden_output = tanh(hidden_input)
final_output = tanh(np.dot(hidden_output, W_output))

print("Predicciones finales (tanh):")
print(final_output)

# Convertimos a -1 y 1 para comparar con y
predictions = np.where(final_output >= 0, 1, -1)
print("Predicciones finales (-1/1):")
print(predictions)
print("Salida esperada:")
print(y)
