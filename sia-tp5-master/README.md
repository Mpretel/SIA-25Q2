# SIA TP5

This repository contains the implementation for the fifth practical assignment (TP5) of the **Intelligent Systems (SIA)** course, focused on **Deep Learning**.  

It includes implementations of an **Autoencoder** applied to the dataset `font.h` and a **Variational Autoencoder (VAE)** applied to the dataset `emojis-x1-32x32`.

Link to presentation: [SIA TP5 Presentation](https://docs.google.com/presentation/d/15e1gVpKAW40LqRVmYUw9BkpukomsZvxLQNi3eNKM6BM/edit?usp=sharing)

> **Note:** This assignment is based on the `MLP` class implemented in **TP3**.  
> Some experiments reuse and extend the methods developed there.

---

## Contents

- `ej1.py`: Implementation of an **Autoencoder**, in order to represent the binary characters of the `font.h` file into a 2D latent space, and a **Denoising Autoencoder** over the same dataset.  
- `ej2.py`: Extension of the **Autoencoder** to a **VAE**, in order to solve the representation into a latent space of an emoji dataset (`emojis-x1-32x32`).
- 
- `constants1.py`: Hyperparameters for the **Autoencoder** on `ej1.py`.
- `constants2.py`: Hyperparameters for the **VAE** on `ej2.py`.

- `font.h`: Binary characters (32 7x5 patterns).
- `load_fonts.py`: Preprocessing of the `font.h` file.

- `emojis-x1-32x32`: Folder with emoji PNG files (50 32x32 RGB emojis).
- `load_emojis.py`: Preprocessing of the files in `emojis-x1-32x32`.

Each exercise includes visualization scripts for inspecting the learned representations.

---

## Requirements

- **Python 3.8+**
- See `requirements.txt` for dependencies  

---

## Usage

Each exercise is implemented as an independent Python script.  
Run them directly from the command line or an IDE (e.g., VS Code, PyCharm, Jupyter).

---

### 🧩 Exercise 1 — Autoencoder (& Denoising Autoencoder)

The file **`ej1.py`** trains a **fully-connected autoencoder** to reconstruct 7×5 character patterns (35 pixels), with optional training as a **denoising autoencoder**.

#### Main steps:
1. **Load the 7×5 character dataset** (`data`, `labels`).
2. **Build two MLPs**:
   - **Encoder**: maps 35 inputs → `LATENT_DIM`.
   - **Decoder**: maps the latent code back to 35 output pixels.  
     Both use implicit bias by concatenating a column of ones.
3. **Preprocessing**:
   - Optional scaling to **[-1, 1]** for centered activation functions.
   - Optional **denoising**: random masking with probability `NOISE_P`.
4. **Training loop**:
   - Mini-batch shuffling.
   - Forward pass: encode → decode.
   - Backward pass: decoder gradient → encoder gradient.
   - Loss: **mean squared error (MSE)**.
   - **Early stopping** based on best epoch loss.
5. **Visualization**:
   - Noisy characters (if denoising is enabled).
   - Reconstructions and thresholded binary outputs.

**Result:**  
The model learns compact latent codes and can reconstruct characters even from noisy inputs (denoising autoencoder).

---

### 😘​ Exercise 2 — Variational Autoencoder (VAE)

The file **`ej2.py`** implements a **Variational Autoencoder**, a generative neural model that learns a continuous latent distribution over the character patterns.

#### Main steps:
1. **Probabilistic encoder**:
   - Outputs the latent mean \( \mu(x) \) and log-variance \( \log \sigma^2(x) \).
   - Uses the **reparameterization trick**  
     \[
     z = \mu + \sigma \odot \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, I)
     \]
     to sample latent vectors while keeping gradients.
2. **Decoder**:
   - Maps latent vectors \( z \) back to the 35-pixel reconstructed pattern.
3. **VAE loss**:
   Combines:
   - **Reconstruction loss** (BCE), and  
   - **KL divergence**  
     \[
     D_{\text{KL}}(q(z|x)\,\|\,\mathcal{N}(0,I)).
     \]
4. **Training**:
   - Joint optimization of reconstruction + latent regularization.
5. **Visualization**:
   - Reconstructions.
   - Latent samples (generated characters).
   - nD latent scatterplot (for latent dimension n).

**Result:**  
The VAE learns a structured latent space that enables interpolation and generation of new character patterns.

### ⚙️ Hyperparameters — `constants1.py` and `constants2.py`

These files define the hyperparameters used for the **Autoencoder** (`constants1.py`) and for the **VAE** (`constants2.py`).  
They control model architecture, optimization settings, training length, and optional noise injection.

---

### Main hyperparameters

- **`BATCH_SIZE`**  
  Size of each mini-batch for training.

- **`LR`**  
  Learning rate used by the optimizer.

- **`ACT_FUNC`** ∈ `{sigmoid, tanh, softsign}`  
  Activation function applied in the MLP layers.

- **`OPTIMIZER`** ∈ `{adam, gd, momentum}`  
  Optimization algorithm:
  - **adam** → adaptive learning  
  - **gd** → gradient descent  
  - **momentum** → accelerated gradient descent

- **`HIDDEN`**  
  Number of hidden units in encoder and decoder.

- **`LATENT_DIM`**  
  Size of the bottleneck latent representation.

- **`EPOCHS`**  
  Maximum number of epochs.

- **`PRINT_EVERY`**  
  Frequency (in epochs) of loss updates.

- **`EARLY_STOPPING_PATIENCE`**  
  Very large patience → effectively disables early stopping unless no improvement for a very long time.

### 🔧 Denoising-specific (only in `constants1.py`)
- **`USE_DENOISING`** ∈ `{TRUE, FALSE}`  
  Enables the denoising autoencoder mode.

- **`NOISE_P`** ∈ `[0, 1]`   
  Probability of masking input pixels during training.

---
