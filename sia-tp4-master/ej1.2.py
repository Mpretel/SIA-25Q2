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
        self.learning_rate = learning_rate
        self.history = []  # para guardar error por época
    
    # Predict
    def predict(self, x):
        return np.dot(self.weights, x)

    # Train the perceptron
    def train(self, X, epochs, w_pca=None):
        # For the fixed number of epochs:
        for epoch in range(epochs):
            # For each training example in the dataset
            for xi in X:
                #calcular el output: y = inner(xi, w )
                yi = self.predict(xi)
                # Update the weights: w += η ∗ y ∗ (xi − y ∗ w)
                self.weights = [w_j + self.learning_rate * yi * (xi_j - yi * w_j) for w_j, xi_j in zip(self.weights, xi)]
            
            # calcular error respecto a PCA si se pasa
            if w_pca is not None:
                cos_sim = np.dot(self.weights, w_pca) / (np.linalg.norm(self.weights) * np.linalg.norm(w_pca))
                error = 1 - abs(cos_sim)
                self.history.append(error)

        return self.weights
    

# 1. Cargar y preparar datos
dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(dir, "europe.csv"))
countries = df["Country"]
X = df.drop(columns=["Country"])

# Estandarizar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
w_pca = pca.components_[0] # primer componente principal
print("PC1 Weights (PCA):", w_pca)

plt.figure(figsize=(8,4))
plt.bar(X.columns, w_pca, color='skyblue')
plt.ylabel("Peso en PC1")
plt.title("Contribución de cada variable a PC1 (PCA)")
plt.axhline(0, color='gray', linewidth=0.8)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Entrenar Oja
perceptron = Perceptron(n_inputs=X_scaled.shape[1])
w = perceptron.train(X_scaled, epochs=150, w_pca=w_pca)

print("PC1 Weights (Oja):", w)

plt.figure(figsize=(8,4))
plt.bar(X.columns, w, color='tomato')
plt.ylabel("Peso en PC1")
plt.title("Contribución de cada variable a PC1 (Oja)")
plt.axhline(0, color='gray', linewidth=0.8)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
plt.plot(perceptron.history)
plt.xlabel("Épocas")
plt.ylabel("Error de dirección (1 - |cos(θ)|)")
plt.title("Evolución del error entre pesos de Oja y PCA")
plt.show()


# Crear DataFrame con los pesos de Oja y PCA
compare_df = pd.DataFrame({
    'Oja': w,
    'PCA': w_pca
}, index=X.columns)

# Gráfico comparativo
plt.figure(figsize=(8, 4))
bar_width = 0.35
x = np.arange(len(compare_df))

plt.bar(x - bar_width/2, compare_df['Oja'], width=bar_width, label='Oja', color='tomato')
plt.bar(x + bar_width/2, compare_df['PCA'], width=bar_width, label='PCA', color='skyblue')

plt.axhline(0, color='gray', linewidth=0.8)
plt.xticks(x, compare_df.index, rotation=45)
plt.ylabel("Peso en PC1")
plt.title("Comparación de contribución por variable (Oja vs PCA)")
plt.legend()
plt.tight_layout()
plt.show()



cosine_history = [1 - e for e in perceptron.history]
plt.figure(figsize=(8,4))
plt.plot(cosine_history)
plt.xlabel("Épocas")
plt.ylabel("Similitud coseno (|cos θ|)")
plt.title("Alineamiento de los pesos con la PC1")
plt.show()
