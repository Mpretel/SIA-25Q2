# SIA TP3

This repository contains the implementation for the third practical assignment (TP3) of the Intelligent Systems course (SIA) about simple and multi-layer perceptrons.

Link to presentation: [SIA TP3 Presentation](https://docs.google.com/presentation/d/1iikkfmx37o4BF3Vm7cpd6B7aibN5tZ_N1QrC-QfF7Fs/edit?usp=sharing)

## Contents

- Code for the simple perceptron (with different activation functions) and the multi-layer perceptron trained with backpropagation (with different activation functions and optimizers)
- Scripts for training, testing, and evaluating models for each excercise

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

> Note: TensorFlow is only required to import the MNIST dataset.

---

## Usage

This repository includes two main models:  
1. **Simple Perceptron (SP)** – for linearly separable or continuous problems.  
2. **Multi-Layer Perceptron (MLP)** – for non-linear problems using backpropagation.

---

### Simple Perceptron (SP)

The `Perceptron` class implements a single neuron that can learn simple relationships depending on the **activation function** used.

#### Example: Using a Step Activation (Linearly Separable Problems)

If you want to solve problems like AND or OR, you can use the `step` activation function.  
This perceptron can only separate **linearly separable** classes.

- Use when: The data can be separated by a straight line (e.g., AND, OR).
- Cannot solve: Non-linear problems such as XOR.

```python
from perceptron import Perceptron
import numpy as np

# Training data for AND logic
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([-1, -1, -1, 1])

# Normalization bounds (not used with step)
X_min, X_max = X.min(axis=0), X.max(axis=0)
y_min, y_max = y.min(), y.max()

# Create perceptron with step activation
p = Perceptron(n_inputs=2, X_min=X_min, X_max=X_max, y_min=y_min, y_max=y_max,
               activation_function='step', learning_rate=0.1)

# Train for up to 20 epochs
errors, _ = p.train(X, y, epochs=20)

# Predict results
for xi in X:
    y_pred, _ = p.predict(xi)
    print(f"{xi} -> {y_pred}")
```

#### Example: Continuous Perceptron (Linear, Sigmoid, or Tanh)

For regression-like or non-binary problems, you can use a continuous activation:
linear, sigmoid, or tanh.
In this case, inputs and outputs are automatically normalized based on the chosen function.

```python
from perceptron import Perceptron
import numpy as np

# Load or generate data
X = np.linspace(-5, 5, 20).reshape(-1, 1)
y = 0.5 * X + 1.0  # Linear function

X_min, X_max = X.min(axis=0), X.max(axis=0)
y_min, y_max = y.min(), y.max()

# Create a perceptron with 'tanh' activation
p = Perceptron(
    n_inputs=1,
    X_min=X_min, X_max=X_max,
    y_min=y_min, y_max=y_max,
    activation_function='tanh',
    beta=2.0,
    learning_rate=0.01
)

errors, _ = p.train(X, y, epochs=200, epsilon=0.001)

# Predict new values
for xi in X:
    y_pred, _ = p.predict(xi)
    print(f"x = {xi[0]:.2f} -> y_pred = {y_pred:.3f}")
```

**Key Parameters**

Parameter	Description	Effect
learning_rate:	Step size for updates --> Higher → faster learning but less stable
beta:	Slope factor (for sigmoid/tanh) --> Higher → sharper transition
epochs:	Max training iterations	--> Increase if not converging
epsilon:	Error threshold for stopping --> Stops early if total error < ε

**Activation Function Summary**

Function	Output Range	Suitable for	Notes
step: {-1, 1}	--> Binary, linearly separable tasks --> No gradient, uses sign threshold
linear: (-∞, ∞)	--> Regression	--> No nonlinearity
sigmoid: (0, 1)	--> Continuous outputs	--> Smooth, can saturate near 0/1
tanh: (-1, 1) --> Continuous outputs	--> Symmetric, good for centered data


### Multi-Layer Perceptron (MLP)

The `MLP` class implements a feedforward neural network with one hidden layer, trained by backpropagation.
It supports multiple optimizers and activation functions.

#### Example: Solving XOR with MLP

```python
from mlp import MLP
import numpy as np

# XOR problem (non-linearly separable)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Create an MLP with 2 input, 4 hidden, and 1 output neuron
mlp = MLP(
    n_input=2,
    n_hidden=4,
    n_output=1,
    activation_function='tanh',
    learning_rate=0.1,
    optimizer='adam'
)

# Train the model
loss_history = mlp.train(X, y, epochs=1000, epsilon=0.001)

# Predict
predictions = mlp.predict(X, method='binary')
print("Predictions:", predictions.flatten())
```

**Main Parameters**

Parameter	Description	Effect
n_input, n_hidden, n_output	Network architecture	More hidden neurons → higher capacity
activation_function	'tanh' or 'sigmoid'	Controls nonlinearity and range
learning_rate	Step size for updates	Higher → faster but less stable
optimizer	'gd', 'momentum', 'adam'	Choice of weight update method
epochs	Max iterations	Higher → longer training
epsilon	Early stopping threshold	Stops when loss < ε
batch_size	None (batch), 1 (online), or N (mini-batch)	Controls update granularity

**Optimizers**

Optimizer	Description	When to use
gd:	Classic gradient descent --> Simple, slower convergence
momentum:	Adds inertia to updates --> Helps escape local minima
adam:	Adaptive learning rates
