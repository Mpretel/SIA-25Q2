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
        return activation_function(z, method='step')

    def train(self, X, y, epochs=20, plot_progress=False):
        for epoch in range(epochs):
            total_error = 0
            # For each training example in the dataset
            for xi, yi in zip(X, y):
                # Prediction
                y_pred = self.predict(xi)
                error = yi - y_pred
                total_error += abs(error)

                # Learning Rule
                self.weights = [w + self.learning_rate * error * xi_j
                                for w, xi_j in zip(self.weights, xi)]
                self.bias += self.learning_rate * error

            print(f"Epoch {epoch+1} - Errors: {total_error}")

            # Plot boundary after each epoch
            if plot_progress:
                plt.clf()  # limpiar figura
                self.plot_decision_boundary(X, y, f"Epoch {epoch+1}")
                plt.pause(0.5)  # pequeña pausa para actualizar

        if plot_progress:
            plt.show()

    def plot_decision_boundary(self, X, y, title=""):
        # Graficar puntos
        for xi, yi in zip(X, y):
            if yi == 1:
                plt.scatter(xi[0], xi[1], color="blue", marker="o", label="1" if "1" not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(xi[0], xi[1], color="red", marker="x", label="-1" if "-1" not in plt.gca().get_legend_handles_labels()[1] else "")

        # Graficar frontera de decisión
        x_vals = [-1.5, 1.5]
        if self.weights[1] != 0:  # evitar división por cero
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
p_and.train(X_and, y_and, epochs=10, plot_progress=True)
print("AND Predictions:")
for xi in X_and:
    print(f"{xi} -> {p_and.predict(xi)}")
"""
La función AND es linealmente separable. Es decir que existe una recta (hiperplano en 2D) que separa las salidas -1 de la salida +1.
Por lo tanto, el perceptrón simple sí puede aprender esta función.
"""
# XOR function
p_xor = Perceptron(n_inputs=2, learning_rate=0.1)
p_xor.train(X_xor, y_xor, epochs=10, plot_progress=True)
print("XOR Predictions:")
for xi in X_xor:
    print(f"{xi} -> {p_xor.predict(xi)}")
"""
La función XOR no es linealmente separable. No existe una recta (hiperplano en 2D) que separe las salidas -1 de la salida +1.
Por lo tanto, el perceptrón simple no puede aprender esta función.
"""
