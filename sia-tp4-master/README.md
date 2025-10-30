# SIA TP4

This repository contains the implementation for the fourth practical assignment (TP4) of the **Intelligent Systems (SIA)** course, focused on **Unsupervised Learning**.  

It includes implementations of **Kohonen Self-Organizing Maps (SOM)**, **Oja’s rule networks**, and **Hopfield associative memory**, applied to the dataset `europe.csv`.

Link to presentation: [SIA TP4 Presentation](https://docs.google.com/presentation/d/1g9YFS_u_gvaAIalk_4Mgr0jAP2zabHK-A_L9CD_7OYA/edit?usp=sharing)

> **Note:** This assignment is based on the `Perceptron` class implemented in **TP3**.  
> Some experiments reuse and extend the methods developed there.

---

## Contents

- `ej1.1.py`: Implementation of a **Kohonen Self-Organizing Map (SOM)** to cluster European countries based on socioeconomic indicators.  
- `ej1.2.py`: Implementation of a **neural network with Oja’s rule** for principal component extraction.  
- `ej2.py`: Implementation of a **Hopfield network** for pattern storage and retrieval.
- `ejPCA.ipynb`: **PCA** analysis using `scikit-learn`.

Each exercise includes visualization scripts for inspecting the learned representations and cluster structures.

---

## Requirements

- **Python 3.8+**
- See `requirements.txt` for dependencies  

> Note: `scikit-learn` is only used for data preprocessing and PCA visualization.

---

## Dataset

The file `europe.csv` contains socioeconomic indicators for European countries.  
It includes one row per country and several normalized numerical features (e.g., GDP, unemployment, industrial output).  
The column `Country` is used for labeling during visualization.

---

## Usage

Each exercise is implemented as an independent Python script.  
Run them directly from the command line or an IDE (e.g., VS Code, PyCharm, Jupyter).

---

### 🧭 Exercise 1.1 — Kohonen Self-Organizing Map (SOM)

The file `ej1.1.py` trains a **4×4 SOM** that groups European countries based on their similarities.

#### Main steps:
1. Load and standardize the dataset (`StandardScaler`).
2. Initialize a 4×4 neuron grid with random or sample-based weights.
3. Iteratively train the map using a Gaussian neighborhood and a decaying learning rate.
4. Assign each country to its Best Matching Unit (BMU).
5. Visualize:
   - Clustering of countries in PCA space.
   - U-Matrix (neighbor distance map).
   - Variable maps for each feature.

### 🧮 Exercise 1.2 — Oja’s Rule

The file `ej1.2.py` implements a simple neural model that learns the **first principal component** of the input data using **Oja’s learning rule**:

\[
\Delta w = \eta (y x - y^2 w)
\]

where \( y = w^T x \).

Useful for **dimensionality reduction** and **unsupervised feature extraction**.

### 🧠 Exercise 2 — Hopfield Network

The file `ej2.py` implements a Hopfield associative memory capable of storing and recalling binary patterns.
The model uses Hebbian learning to define a stable energy landscape.
