import numpy as np
import matplotlib.pyplot as plt
import random
from MLP import MLP, activation_functions
from constants1 import *
from load_fonts import data, labels

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# Graficar patrones
def show_pattern(vec35, ax, title=None):
    mat = vec35.reshape(7, 5)
    ax.imshow(mat, cmap='gray_r', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)

n_input = data.shape[1]  # 35

#  Autoencoder usando MLP
encoder = MLP(n_input=n_input + 1, n_hidden=HIDDEN, n_output=LATENT_DIM,
              activation_function=ACT_FUNC, learning_rate=LR, optimizer=OPTIMIZER)
decoder = MLP(n_input=LATENT_DIM + 1, n_hidden=HIDDEN, n_output=n_input,
              activation_function=ACT_FUNC, learning_rate=LR, optimizer=OPTIMIZER)



if ACT_FUNC == 'tanh' or ACT_FUNC == 'softsign':
    data = data * 2.0 - 1.0  # Escalar datos de [0, 1] a [-1, 1]

thd = activation_functions[ACT_FUNC][3] # umbral de activación


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

n = len(data_noisy)
plt.figure(figsize=(12, 6))
for i in range(n):
    ax = plt.subplot(4, 8, i + 1)
    title = f"{labels[i]}"
    show_pattern(data_noisy[i], ax, title)
plt.suptitle(f"Caracteres con ruido (p={NOISE_P})", fontsize=12)
plt.tight_layout()
plt.show()



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

noise_levels = np.arange(0, 1, 0.05)

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
    plt.suptitle(f"Reconstrucciones con noise={round(NOISE_P, 2)} (Error total={errors.sum()})", fontsize=14)

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

