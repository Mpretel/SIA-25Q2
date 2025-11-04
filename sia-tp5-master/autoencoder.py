import numpy as np

# Funciones de activación
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

# Multi-Layer Perceptron Class
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

    def predict(self, X, method='binary'):
        """Predicción binaria"""
        output = self.forward(X)

        if method == 'binary':
            if self.range == (-1, 1):
                predictions = np.where(output >= self.threshold, 1, -1)
            elif self.range == (0, 1):
                predictions = np.where(output >= self.threshold, 1, 0)
        elif method == 'multiclass':
            predictions = np.argmax(output, axis=1)

        return predictions


# Autoencoder Class
class Autoencoder:
    def __init__(self, n_input=35, n_hidden=16, n_latent=2, activation='tanh', learning_rate=0.1, optimizer='adam'):
        # Codificador
        self.encoder = MLP(n_input, n_hidden, n_latent,
                           activation_function=activation,
                           learning_rate=learning_rate,
                           optimizer=optimizer)
        # Decodificador
        self.decoder = MLP(n_latent, n_hidden, n_input,
                           activation_function=activation,
                           learning_rate=learning_rate,
                           optimizer=optimizer)

    def forward(self, X):
        """Codifica y decodifica"""
        latent = self.encoder.forward(X)
        reconstructed = self.decoder.forward(latent)
        return reconstructed, latent

    def train(self, X, epochs=1000, epsilon=0.0, batch_size=None):
        """Entrena para minimizar el error de reconstrucción"""
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        loss_history = []
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                Xi = X_shuffled[start:end]

                # Forward
                reconstructed, latent = self.forward(Xi)

                # Backward: el target es la propia entrada
                loss_decoder = self.decoder.backward(latent, Xi, reconstructed)

                # Retropropagar error hacia el codificador
                error_latent = (Xi - reconstructed) @ self.decoder.W_output.T * self.encoder.activation_deriv(self.encoder.final_input)
                self.encoder.W_output += self.encoder.learning_rate * self.encoder.hidden_output.T.dot(error_latent)
                self.encoder.W_hidden += self.encoder.learning_rate * Xi.T.dot(error_latent.dot(self.encoder.W_output.T) * self.encoder.activation_deriv(self.encoder.hidden_input))

            loss_history.append(loss_decoder)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss={loss_decoder:.4f}")

            if loss_decoder <= epsilon:
                print(f"Early stopping at epoch {epoch}, Loss={loss_decoder:.4f}")
                return loss_history

        return loss_history

    def encode(self, X):
        """Obtiene la representación latente"""
        return self.encoder.forward(X)

    def decode(self, Z):
        """Reconstruye a partir del espacio latente"""
        return self.decoder.forward(Z)