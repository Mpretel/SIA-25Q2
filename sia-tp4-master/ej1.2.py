import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.decomposition import PCA


# fijar semilla para reproducibilidad
SEED = 10
np.random.seed(SEED)
random.seed(SEED)



# Perceptron Class
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.001):
        # Initialize weights and bias to small random values
        self.weights = [random.uniform(0, 1) for _ in range(n_inputs)]
        # Set learning rate
        self.learning_rate = learning_rate
    
    # Activation Function
    def activation_function(self, z):
        tita = z
        return tita
    
    # Predict
    def predict(self, x):
        # Calculate the weighted sum
        z = sum(w * xi for w, xi in zip(self.weights, x))
        # Compute activation given by the activation function
        y_pred = self.activation_function(z)
        return y_pred
    
    # Train the perceptron
    def train(self, X, epochs):

        # For the fixed number of epochs:
        for epoch in range(epochs):
            # For each training example in the dataset
            for i, xi in enumerate(X):
                yi = self.predict(xi)     #calcular el output: y = inner(xi, w )

                # Update the weights: w += η ∗ y ∗ (xi − y ∗ w)
                self.weights = [w_j + self.learning_rate * yi * (xi_j - yi * w_j) for w_j, xi_j in zip(self.weights, xi)]

        return self.weights
    





# 1. Cargar y preparar datos
dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(dir, "europe.csv"))
countries = df["Country"]
X = df.drop(columns=["Country"])

# Estandarizar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


perceptron = Perceptron(n_inputs=X_scaled.shape[1])
w = perceptron.train(X_scaled, epochs=1000)

print(w)



# Reducir a 2D con PCA solo para visualización
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

loadings = pca.components_.T 
print(loadings)