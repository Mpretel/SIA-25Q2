import numpy as np
import matplotlib.pyplot as plt
import random
from constants1 import *
from data import data, labels


SEED = 42
np.random.seed(SEED)
random.seed(SEED)


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
    
# Graficar patrones
def show_pattern(vec35, ax, title=None):
    mat = vec35.reshape(7, 5)
    ax.imshow(mat, cmap='gray_r', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


#  Autoencoder usando MLP
encoder = MLP(n_input=35 + 1, n_hidden=HIDDEN, n_output=LATENT_DIM,
              activation_function=ACT_FUNC, learning_rate=LR, optimizer=OPTIMIZER)
decoder = MLP(n_input=LATENT_DIM + 1, n_hidden=HIDDEN, n_output=35,
              activation_function=ACT_FUNC, learning_rate=LR, optimizer=OPTIMIZER)



if ACT_FUNC == 'tanh' or ACT_FUNC == 'softsign':
    data = data * 2.0 - 1.0  # Escalar datos de [0, 1] a [-1, 1]

thd = activation_functions[ACT_FUNC][3] # umbral de activación

# ====================================================
# Denoising (antes del entrenamiento)
# en caso de usar cambiar data por data_noisy en el entrenamiento

# if USE_DENOISING:
#     noisy_list = []
#     clean_list = []

#     for x in data:
#         for _ in range(N_SAMPLES):
#             mask = np.random.rand(*x.shape) > NOISE_P

#             if ACT_FUNC in ["tanh", "softsign"]:
#                 x_noisy = x * mask + (-1.0) * (~mask)
#             else:
#                 x_noisy = x * mask
            
#             noisy_list.append(x_noisy)
#             clean_list.append(x)       # Target limpio

#     data_noisy = np.array(noisy_list)
#     data_clean = np.array(clean_list)
# else:
#     data_noisy = data
#     data_clean = data

# # Muestro un caracter ruidoso de cada uno
# rows, cols = 4, 8
# N = rows * cols   # 32 muestras

# plt.figure(figsize=(cols*1.5, rows*1.5))
# plt.suptitle(f"Caracteres ruidosos generados (ruido = {NOISE_P})", fontsize=14)

# for i in range(N):
#     ax = plt.subplot(rows, cols, i + 1)
#     show_pattern(data_noisy[i+(N_SAMPLES-1)*i], ax, title=f"{labels[i]}")
    
# plt.tight_layout()
# plt.show()
#============================================================

#  Entrenamiento con bias implícito
best_loss = np.inf
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    idx = np.random.permutation(len(data))
    batches = [idx[i:i + BATCH_SIZE] for i in range(0, len(idx), BATCH_SIZE)]
    epoch_loss = 0.0

    for batch in batches:
        x = data[batch]
        
        # Denoising (durante el entrenamiento)
        if USE_DENOISING:
            mask = np.random.rand(*x.shape) > NOISE_P
            if ACT_FUNC == 'tanh' or ACT_FUNC == 'softsign':
                x_noisy = x * mask + (-1.0) * (~mask)
            else:
                x_noisy = x * mask
        else:
            x_noisy = x
        

        # --- Agregar columna de 1s (bias implícito) ---
        x_noisy_bias = np.hstack([x_noisy, np.ones((x_noisy.shape[0], 1))])

        # === Forward ===
        z = encoder.forward(x_noisy_bias)

        # Agregar bias al espacio latente
        z_bias = np.hstack([z, np.ones((z.shape[0], 1))])

        recon = decoder.forward(z_bias)

        # === Backward ===
        decoder_grad_z = decoder.backward(z_bias, y=x, output=recon)
        decoder_grad_z = decoder_grad_z[:, :-1]   # quitar gradiente del bias antes del encoder
        encoder.backward(x_noisy_bias, grad_output=decoder_grad_z)

        # === Pérdida ===
        loss = np.mean((x - recon)**2)
        epoch_loss += loss * len(batch)

    # --- Promedio por época ---
    epoch_loss /= len(data)

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Loss={epoch_loss:.6f}")

    # --- Early stopping ---
    if epoch_loss < best_loss - 1e-6:
        best_loss = epoch_loss
        best_encoder = encoder
        best_decoder = decoder
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= EARLY_STOPPING_PATIENCE:
        print("Early stopping.")
        break

encoder, decoder = best_encoder, best_decoder


# --- Denoising opcional ---
if USE_DENOISING:
    mask = np.random.rand(*data.shape) > NOISE_P
    if ACT_FUNC == 'tanh' or ACT_FUNC == 'softsign':
        data_noisy = data * mask + (-1.0) * (~mask)
    else:
        data_noisy = data * mask
else:
    data_noisy = data   


# Agregar bias en la entrada
data_noisy_bias = np.hstack([data_noisy, np.ones((data_noisy.shape[0], 1))])
z_np = encoder.predict(data_noisy_bias)

# Agregar bias al espacio latente antes del decoder
z_bias = np.hstack([z_np, np.ones((z_np.shape[0], 1))])
probs = decoder.predict(z_bias)

### --- Reconstrucciones y errores --- ###
recons = (probs >= thd).astype(int)
if ACT_FUNC == 'tanh' or ACT_FUNC == 'softsign':
    recons = recons * 2.0 - 1.0  # Escalar datos de [0, 1] a [-1, 1]

errors = np.sum(recons != data, axis=1)

print("Errores por muestra:", errors)
print("Errores promedio:", errors.mean(), "máx:", errors.max())
print("Error total:", errors.sum())

#  Visualización del espacio latente
plt.figure(figsize=(7, 7))
plt.scatter(z_np[:, 0], z_np[:, 1], s=60, color='steelblue')
for i, (xx, yy) in enumerate(z_np):
    plt.text(xx + 0.02, yy + 0.02, labels[i], fontsize=10, fontfamily='monospace')
plt.title("Espacio latente 2D del Autoencoder (MLP)")
plt.xlabel("z₁"); plt.ylabel("z₂")
plt.grid(True)
plt.show()


n = len(data)
plt.figure(figsize=(12, 6))
for i in range(n):
    ax = plt.subplot(4, 8, i + 1)
    title = f"{labels[i]}\nerr={errors[i]}"
    show_pattern(recons[i], ax, title)
plt.suptitle("Reconstrucciones con error por patrón", fontsize=12)
plt.tight_layout()
plt.show()


################################################################
#Distintos niveles de ruido
#Ver qué pasa que en 0.3 no da 0

noise_levels = np.arange(0, 1, 0.1)

all_errors = {} 
all_recons = {} 

for NOISE_P in noise_levels:
    print(f"\n=== Noise = {NOISE_P} ===")

    # Generar ruido 
    mask = np.random.rand(*data.shape) > NOISE_P
    
    if ACT_FUNC in ["tanh", "softsign"]:
        data_noisy = data * mask + (-1.0) * (~mask)
    else:
        data_noisy = data * mask

    # Forward
    data_noisy_bias = np.hstack([data_noisy, np.ones((data_noisy.shape[0], 1))])
    z_np = encoder.predict(data_noisy_bias)

    z_bias = np.hstack([z_np, np.ones((z_np.shape[0], 1))])
    probs = decoder.predict(z_bias)

    # Reconstrucción
    recons = (probs >= thd).astype(int)
    if ACT_FUNC in ["tanh", "softsign"]:
        recons = recons * 2.0 - 1.0

    # Errores
    errors = np.sum(recons != data, axis=1)

    all_errors[NOISE_P] = {
        "errors": errors,
        "mean": errors.mean(),
        "max": errors.max(),
        "total": errors.sum()
    }

    all_recons[NOISE_P] = (data_noisy, recons)

    print(f"Error medio: {errors.mean():.2f}, máx: {errors.max()}, total: {errors.sum()}")

for NOISE_P in noise_levels:

    data_noisy, recons = all_recons[NOISE_P]
    errors = all_errors[NOISE_P]["errors"]

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Reconstrucciones con noise={NOISE_P} (Error total={errors.sum()})", fontsize=14)

    n = len(data_noisy)

    for i in range(n):
        ax = plt.subplot(4, 8, i + 1)
        title = f"{labels[i]}\nerr={errors[i]}"
        show_pattern(recons[i], ax, title)

    plt.tight_layout()
    plt.show()

# Errores vs noise levels
plt.figure(figsize=(12,6), dpi=120)
plt.scatter(noise_levels,
            [all_errors[nl]["total"] for nl in noise_levels], color = "#5557a3ff")
plt.xlabel("Nivel de ruido")
plt.ylabel("Errores totales")
plt.title("Errores totales vs niveles de ruido")
plt.grid()
plt.show()

################################################################

"""
### --- Grid del espacio latente (decodificación) --- ###
zmin = z_np.min(axis=0) - 0.5
zmax = z_np.max(axis=0) + 0.5
nx, ny = 8, 8

grid = np.array([
    [zmin[0] + (zmax[0]-zmin[0])*j/(nx-1),
     zmin[1] + (zmax[1]-zmin[1])*i/(ny-1)]
    for i in range(ny) for j in range(nx)
])

# Agregar bias
grid_bias = np.hstack([grid, np.ones((grid.shape[0], 1))])

# Decodificar
gen_probs = decoder.predict(grid_bias)
gen = (gen_probs >= thd).astype(int)

if ACT_FUNC in ['tanh', 'softsign']:
    gen = gen * 2 - 1

# Invertir filas
gen_grid = gen.reshape(ny, nx, -1)
gen_grid = gen_grid[::-1]  # invertir filas

# Graficar
plt.figure(figsize=(6, 6))
idx = 1
for i in range(ny):
    for j in range(nx):
        ax = plt.subplot(ny, nx, idx)
        show_pattern(gen_grid[i, j], ax)
        idx += 1

plt.suptitle("Grid decode desde el espacio latente (MLP)")
plt.show()



### --- Muestreo cercano --- ###

idx = 3
z0 = z_np[idx]
target_label = labels[idx]

radius = 0.05
samples = 8 

angles = np.linspace(0, 2*np.pi, samples, endpoint=False)
points = np.vstack([
    z0 + radius*np.array([np.cos(a), np.sin(a)])
    for a in angles
])

# Bias
points_bias = np.hstack([points, np.ones((points.shape[0], 1))])
probs = decoder.predict(points_bias)

samples_bin = (probs >= thd).astype(int)
if ACT_FUNC in ["tanh", "softsign"]:
    samples_bin = samples_bin * 2 - 1


# Figura objetivo
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
ax.set_title(f"Objetivo: '{target_label}'")

show_pattern(recons[idx], ax)

plt.tight_layout()
plt.show()


# Figura samples
half = samples // 2

fig, axes = plt.subplots(2, half, figsize=(2.2*half, 4))
fig.suptitle("Muestras generadas alrededor del objetivo", fontsize=12)

# --- Primera fila ---
for i in range(half):
    ax = axes[0, i]
    show_pattern(samples_bin[i], ax, title=f"θ={int(angles[i]*180/np.pi)}°")

# --- Segunda fila ---
for i in range(half, samples):
    ax = axes[1, i-half]
    show_pattern(samples_bin[i], ax, title=f"θ={int(angles[i]*180/np.pi)}°")

plt.tight_layout()
plt.show()


# Visualización en el espacio latente
plt.figure(figsize=(6, 6))
plt.scatter(z_np[:,0], z_np[:,1], color='lightgray', s=40)
plt.scatter(points[:,0], points[:,1], color='crimson', s=70)
plt.scatter([z0[0]], [z0[1]], color='blue', s=120)

for p in points:
    plt.plot([z0[0], p[0]], [z0[1], p[1]], "k--", alpha=0.3)

plt.title(f"Muestreo alrededor de '{target_label}'")
plt.xlabel("z₁"); plt.ylabel("z₂")
plt.grid(True)
plt.show()


### --- Grid search --- ###

#  Autoencoder usando MLP
encoder = MLP(n_input=35 + 1, n_hidden=HIDDEN, n_output=LATENT_DIM,
              activation_function=ACT_FUNC, learning_rate=LR, optimizer=OPTIMIZER)
decoder = MLP(n_input=LATENT_DIM + 1, n_hidden=HIDDEN, n_output=35,
              activation_function=ACT_FUNC, learning_rate=LR, optimizer=OPTIMIZER)



if ACT_FUNC == 'tanh' or ACT_FUNC == 'softsign':
    data = data * 2.0 - 1.0  # Escalar datos de [0, 1] a [-1, 1]

thd = activation_functions[ACT_FUNC][3] # umbral de activación



#  Entrenamiento con bias implícito
best_loss = np.inf
no_improve = 0

def train_autoencoder_return_loss(batch_size, hidden_size):

    encoder = MLP(
        n_input=35 + 1,
        n_hidden=hidden_size,
        n_output=LATENT_DIM,
        activation_function=ACT_FUNC,
        learning_rate=LR,
        optimizer=OPTIMIZER
    )
    decoder = MLP(
        n_input=LATENT_DIM + 1,
        n_hidden=hidden_size,
        n_output=35,
        activation_function=ACT_FUNC,
        learning_rate=LR,
        optimizer=OPTIMIZER
    )

    best_loss = np.inf
    no_improve = 0

    loss_curve = []

    for epoch in range(1, EPOCHS + 1):

        idx = np.random.permutation(len(data))
        batches = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
        epoch_loss = 0.0

        for batch in batches:
            x = data[batch]
            x_bias = np.hstack([x, np.ones((x.shape[0], 1))])

            # Forward
            z = encoder.forward(x_bias)
            z_bias = np.hstack([z, np.ones((z.shape[0], 1))])
            recon = decoder.forward(z_bias)

            # Backward
            dec_grad = decoder.backward(z_bias, y=x, output=recon)
            dec_grad = dec_grad[:, :-1]   # quitar bias
            encoder.backward(x_bias, grad_output=dec_grad)

            epoch_loss += np.mean((x - recon)**2) * len(batch)

        epoch_loss /= len(data)
        loss_curve.append(epoch_loss)

        # early stopping
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_encoder = encoder
            best_decoder = decoder
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOPPING_PATIENCE:
            break

    encoder = best_encoder
    decoder = best_decoder

    # --- Evaluación ---
    data_bias = np.hstack([data, np.ones((data.shape[0], 1))])
    z = encoder.predict(data_bias)
    z_bias = np.hstack([z, np.ones((z.shape[0], 1))])
    probs = decoder.predict(z_bias)

    recons = (probs >= thd).astype(int)
    if ACT_FUNC in ["tanh", "softsign"]:
        recons = recons * 2 - 1

    errors_total = np.sum(recons != data)

    return errors_total, loss_curve

batch_sizes  = [8, 16]
hidden_sizes = [16, 64, 128]

heatmap_errors = np.zeros((len(batch_sizes), len(hidden_sizes)))
loss_curves = {}

for i, bs in enumerate(batch_sizes):
    for j, hs in enumerate(hidden_sizes):
        print(f"Entrenando: batch={bs}, hidden={hs}...")
        errors_total, loss_curve = train_autoencoder_return_loss(bs, hs)
        heatmap_errors[i, j] = errors_total
        loss_curves[(bs, hs)] = loss_curve

plt.figure(figsize=(8, 6), dpi=120)
plt.imshow(heatmap_errors, origin="lower", aspect="auto")
plt.colorbar(label="Errores totales")

plt.xticks(range(len(hidden_sizes)), hidden_sizes)
plt.yticks(range(len(batch_sizes)), batch_sizes)

plt.xlabel("Hidden neurons")
plt.ylabel("Batch size")
plt.title("Heatmap: errores totales de reconstrucción")
plt.show()

plt.figure(figsize=(10, 6))

for key, curve in loss_curves.items():
    bs, hs = key
    plt.plot(curve, label=f"bs={bs}, h={hs}", alpha=0.7)

plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curvas de loss por configuración")
plt.legend()
plt.show()

### --- Fin Grid search --- ###
"""