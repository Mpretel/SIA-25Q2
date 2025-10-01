import random
import matplotlib.pyplot as plt



# Perceptron Class
class Perceptron:
    def __init__(self, n, learning_rate=0.1):
        # Initialize weights and bias to small random values
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(n)]
        self.bias = random.uniform(-0.01, 0.01)
        # Set learning rate
        self.learning_rate = learning_rate

    # Activation Function
    def activation_function(self, z):
        return 1 if z >= 0 else -1
        
    def predict(self, x):
        # Calculate the weighted sum
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        # Compute activation given by the activation function
        y_pred = self.activation_function(z)
        return y_pred

    def calculate_error(self, yi, y_pred):
        return yi - y_pred

    def train(self, X, y, epochs=20, epsilon=0.0):
        error_history = []

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))
        plt.ion() 

        # For the fixed number of epochs:
        for epoch in range(epochs):

            # For each training example in the dataset
            for xi, yi in zip(X, y):
                # Prediction
                y_pred = self.predict(xi)

                # Calculate error: y_real - y_pred
                error = self.calculate_error(yi, y_pred)
                
                # Update the weights and bias
                # wi = wi + learning_rate * error * xi_j
                self.weights = [w + self.learning_rate * error * xi_j for w, xi_j in zip(self.weights, xi)]
                # bias = bias + learning_rate * error
                self.bias += self.learning_rate * error

                total_error = sum(abs(self.calculate_error(yi, self.predict(xi))) for xi, yi in zip(X, y))

                print(f"Epoch {epoch+1} - Errors: {total_error}")
                error_history.append(total_error)

                ax1.clear()
                ax2.clear()
                ax1.set_xlim(-2.5, 2.5)
                ax1.set_ylim(-2.5, 2.5)
                self.plot_decision_boundary(X, y, f"Epoch {epoch+1}", ax=ax1)
                ax2.plot(range(1, len(error_history)+1), error_history, marker="o")
                ax2.set_title("Evolución del error por actualización de los pesos")
                ax2.set_xlabel("Actualización")
                ax2.set_ylabel("Error total")
                plt.pause(0.5)

            # Early stopping due to convergence
            if total_error <= epsilon:
                print(f"Converged at epoch {epoch+1} with total_error={total_error}")
                break

        plt.ioff()
        plt.show()
        

    def plot_decision_boundary(self, X, y, title="", ax=None):
        if ax is None:
            ax = plt.gca()

        # Plot data points
        for xi, yi in zip(X, y):
            if yi == 1:
                ax.scatter(xi[0], xi[1], color="blue", marker="o", 
                        label="1" if "1" not in ax.get_legend_handles_labels()[1] else "")
            else:
                ax.scatter(xi[0], xi[1], color="red", marker="x", 
                        label="-1" if "-1" not in ax.get_legend_handles_labels()[1] else "")

        # Plot decision boundary
        x_vals = [-1.5, 1.5]
        if self.weights[1] != 0:
            y_vals = [-(self.weights[0]*x + self.bias)/self.weights[1] for x in x_vals]
            ax.plot(x_vals, y_vals, 'k--', label="Decision boundary")
        
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc='upper right')


# Training Data
# AND function
X_and = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
y_and = [-1, -1, -1, 1]
# XOR function
X_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
y_xor = [1, 1, -1, -1]

"""
for X, y, function_name in zip([X_and, X_xor], [y_and, y_xor], ["AND", "XOR"]):
    plt.figure(figsize=(7,6))
    plt.title(f"{function_name} Function Data Points")
    plt.xlabel("x1")
    plt.ylabel("x2")
    for xi, yi in zip(X, y):
        if yi == 1:
            plt.scatter(xi[0], xi[1], color="blue", marker="o", label="1" if "1" not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(xi[0], xi[1], color="red", marker="x", label="-1" if "-1" not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.show()
  """  

# Train Perceptron 
# AND function
p_and = Perceptron(n=2, learning_rate=0.1)
p_and.train(X_and, y_and, epochs=10, epsilon=0.0)
print("AND Predictions:")
for xi in X_and:
    print(f"{xi} -> {p_and.predict(xi)}")

# XOR function
p_xor = Perceptron(n=2, learning_rate=0.1)
p_xor.train(X_xor, y_xor, epochs=10, epsilon=0.0)
print("XOR Predictions:")
for xi in X_xor:
    print(f"{xi} -> {p_xor.predict(xi)}")
