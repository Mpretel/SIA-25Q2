import random
import numpy as np

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