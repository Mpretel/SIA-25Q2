import numpy as np


#  Activaciones y clase MLP
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def softsign(x):
    return x / (1 + np.abs(x))

def softsign_derivative(x):
    return 1 / (1 + np.abs(x))**2

activation_functions = {
    'tanh': (tanh, tanh_derivative, (-1, 1), 0.0),
    'sigmoid': (sigmoid, sigmoid_derivative, (0, 1), 0.5),
    'softsign': (softsign, softsign_derivative, (-1, 1), 0.0)
}

class MLP:
    def __init__(self, n_input, n_hidden, n_output, activation_function='tanh', learning_rate=0.1, optimizer='gd',
                 beta=0.9, beta1=0.9, beta2=0.999, epsilon_adam=1e-8):
        self.W_hidden = np.random.uniform(-0.1, 0.1, (n_input, n_hidden))
        self.W_output = np.random.uniform(-0.1, 0.1, (n_hidden, n_output))
        self.learning_rate = learning_rate
        self.activation_function_name = activation_function
        self.activation_func, self.activation_deriv, _, _ = activation_functions[activation_function]
        self.optimizer = optimizer
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon_adam = epsilon_adam
        self.iteration = 0
        # Inicialización de momentum/Adam
        self.vW_hidden = np.zeros_like(self.W_hidden)
        self.vW_output = np.zeros_like(self.W_output)
        self.mW_hidden = np.zeros_like(self.W_hidden)
        self.vW_hidden_adam = np.zeros_like(self.W_hidden)
        self.mW_output = np.zeros_like(self.W_output)
        self.vW_output_adam = np.zeros_like(self.W_output)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.W_hidden)
        self.hidden_output = self.activation_func(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.W_output)
        self.final_output = self.activation_func(self.final_input)
        return self.final_output

    def backward(self, X, y=None, output=None, grad_output=None):
        if grad_output is None:
            # backward normal con target
            error = y - output
            d_output = error * self.activation_deriv(self.final_input)
        else:
            # backward con gradiente externo (para el encoder)
            d_output = grad_output * self.activation_deriv(self.final_input)

        d_hidden = d_output.dot(self.W_output.T) * self.activation_deriv(self.hidden_input)
        grad_W_output = self.hidden_output.T.dot(d_output)
        grad_W_hidden = X.T.dot(d_hidden)
        if self.optimizer == 'gd':
            self.W_output += self.learning_rate * grad_W_output
            self.W_hidden += self.learning_rate * grad_W_hidden
        elif self.optimizer == 'momentum':
            self.vW_output = self.beta * self.vW_output + (1 - self.beta) * grad_W_output
            self.vW_hidden = self.beta * self.vW_hidden + (1 - self.beta) * grad_W_hidden
            self.W_output += self.learning_rate * self.vW_output
            self.W_hidden += self.learning_rate * self.vW_hidden
        elif self.optimizer == 'adam':
            self.iteration += 1
            self.mW_output = self.beta1 * self.mW_output + (1 - self.beta1) * grad_W_output
            self.vW_output_adam = self.beta2 * self.vW_output_adam + (1 - self.beta2) * (grad_W_output**2)
            self.mW_hidden = self.beta1 * self.mW_hidden + (1 - self.beta1) * grad_W_hidden
            self.vW_hidden_adam = self.beta2 * self.vW_hidden_adam + (1 - self.beta2) * (grad_W_hidden**2)
            m_hat_out = self.mW_output / (1 - self.beta1**self.iteration)
            v_hat_out = self.vW_output_adam / (1 - self.beta2**self.iteration)
            m_hat_hid = self.mW_hidden / (1 - self.beta1**self.iteration)
            v_hat_hid = self.vW_hidden_adam / (1 - self.beta2**self.iteration)
            self.W_output += self.learning_rate * m_hat_out / (np.sqrt(v_hat_out) + self.epsilon_adam)
            self.W_hidden += self.learning_rate * m_hat_hid / (np.sqrt(v_hat_hid) + self.epsilon_adam)
        
        # gradiente respecto a la entrada de la red
        d_input = d_hidden.dot(self.W_hidden.T)

        return d_input

    def predict(self, X):
        return self.forward(X)
    