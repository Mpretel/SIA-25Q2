import random
import matplotlib.pyplot as plt

# Activation Function
def activation_function(z, method='step'):
    if method == 'step':
        return 1 if z >= 0 else -1

# Perceptron Class
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        # Initialize weights and bias to small random values
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(n_inputs)]
        self.bias = random.uniform(-0.01, 0.01)
        # Set learning rate
        self.learning_rate = learning_rate

    def predict(self, x):
        # Calculate the weighted sum
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        # Compute activation given by the activation function
        y_pred = activation_function(z, method='step')
        return y_pred

    def train(self, X, y, epochs=20, epsilon=0.0, plot_progress=False):
        # For the fixed number of epochs:
        for epoch in range(epochs):
            total_error = 0
            # For each training example in the dataset
            for xi, yi in zip(X, y):
                # Prediction
                y_pred = self.predict(xi)

                # Calculate error: y_real - y_pred
                error = yi - y_pred
                total_error += abs(error)

                # Update the weights and bias
                # wi = wi + learning_rate * error * xi_j
                self.weights = [w + self.learning_rate * error * xi_j for w, xi_j in zip(self.weights, xi)]
                # bias = bias + learning_rate * error
                self.bias += self.learning_rate * error

            print(f"Epoch {epoch+1} - Errors: {total_error}")

            # Early stopping due to convergence
            if total_error <= epsilon:
                print(f"Converged at epoch {epoch+1} with total_error={total_error}")
                break

            # Plot boundary after each epoch
            if plot_progress:
                plt.clf()  # clear figure
                self.plot_decision_boundary(X, y, f"Epoch {epoch+1}")
                plt.pause(0.5)  # small pause to update

        if plot_progress:
            plt.show()

    def plot_decision_boundary(self, X, y, title=""):
        # Plot data points
        for xi, yi in zip(X, y):
            if yi == 1:
                plt.scatter(xi[0], xi[1], color="blue", marker="o", label="1" if "1" not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(xi[0], xi[1], color="red", marker="x", label="-1" if "-1" not in plt.gca().get_legend_handles_labels()[1] else "")

        # Plot decision boundary
        x_vals = [-1.5, 1.5]
        if self.weights[1] != 0:  # avoid division by zero
            y_vals = [-(self.weights[0]*x + self.bias)/self.weights[1] for x in x_vals]
            plt.plot(x_vals, y_vals, 'k--', label="Decision boundary")
        
        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True)


# Training Data
# AND function
X_and = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
y_and = [-1, -1, -1, 1]
# XOR function
X_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
y_xor = [1, 1, -1, -1]


# Train Perceptron 
# AND function
p_and = Perceptron(n_inputs=2, learning_rate=0.1)
p_and.train(X_and, y_and, epochs=10, epsilon=0.0, plot_progress=True)
print("AND Predictions:")
for xi in X_and:
    print(f"{xi} -> {p_and.predict(xi)}")
"""
La función AND es linealmente separable. Es decir que existe una recta (hiperplano en 2D) que separa las salidas -1 de la salida +1.
Por lo tanto, el perceptrón simple sí puede aprender esta función.
"""
# XOR function
p_xor = Perceptron(n_inputs=2, learning_rate=0.1)
p_xor.train(X_xor, y_xor, epochs=10, epsilon=0.0, plot_progress=True)
print("XOR Predictions:")
for xi in X_xor:
    print(f"{xi} -> {p_xor.predict(xi)}")
"""
La función XOR no es linealmente separable. No existe una recta (hiperplano en 2D) que separe las salidas -1 de la salida +1.
Por lo tanto, el perceptrón simple no puede aprender esta función.
"""
