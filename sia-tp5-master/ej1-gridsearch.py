import numpy as np
import matplotlib.pyplot as plt
import random
from MLP import MLP, activation_functions
from constants1 import *
from load_fonts import data, labels


SEED = 42
np.random.seed(SEED)
random.seed(SEED)


n_input = data.shape[1]  # 35


if ACT_FUNC == 'tanh' or ACT_FUNC == 'softsign':
    data = data * 2.0 - 1.0  # Escalar datos de [0, 1] a [-1, 1]

thd = activation_functions[ACT_FUNC][3] # umbral de activación


def train_autoencoder_return_loss(act_func, optimizer, lr, batch_size, hidden_size):

    encoder = MLP(
        n_input=n_input + 1,
        n_hidden=hidden_size,
        n_output=LATENT_DIM,
        activation_function=act_func,
        learning_rate=lr,
        optimizer=optimizer
    )
    decoder = MLP(
        n_input=LATENT_DIM + 1,
        n_hidden=hidden_size,
        n_output=n_input,
        activation_function=act_func,
        learning_rate=lr,
        optimizer=optimizer
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
    if act_func in ["tanh", "softsign"]:
        recons = recons * 2 - 1

    errors_total = np.sum(recons != data)

    return errors_total, loss_curve

# -------------------------------------------
# GRID SEARCH RANKING
# -------------------------------------------
"""
optimizers  = ['gd', 'momentum', 'adam']
lrs = [1e-4, 5e-4, 1e-3]
batch_sizes  = [8, 16, 32]
hidden_sizes = [16, 32, 64]

resultados = []

for i, optimizer in enumerate(optimizers):
    for j, lr in enumerate(lrs):
        for k, batch_size in enumerate(batch_sizes):
            for l, hidden_size in enumerate(hidden_sizes):
                print(f"Entrenando: opt={optimizer}, lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}...")
                errors_total, loss_curve = train_autoencoder_return_loss(ACT_FUNC, optimizer, lr, batch_size, hidden_size)


                # Guardar los resultados
                resultados.append({
                    "optimizer": optimizer,
                    "lr": lr,
                    "batch_size": batch_size,
                    "hidden_size": hidden_size,
                    "error": errors_total
                })


# Ordenar por error (de menor a mayor)
resultados_ordenados = sorted(resultados, key=lambda x: x["error"])

# Imprimir el ranking final
print("\n=== Ranking de configuraciones por error total ===")
for idx, r in enumerate(resultados_ordenados):
    print(f"{idx+1}. error={r['error']:.6f} | opt={r['optimizer']}, lr={r['lr']}, batch={r['batch_size']}, hidden={r['hidden_size']}")
"""

# -------------------------------------------
# GRID SEARCH PLOT HEATMAP Y CURVAS DE LOSS
# -------------------------------------------
        
optimizers  = ['adam']
lrs = [5e-4, 1e-3]
batch_sizes  = [8, 16, 32]
hidden_sizes = [16, 32, 64]

resultados = []

for k, optimizer in enumerate(optimizers):
    for l, lr in enumerate(lrs):
        heatmap_errors = np.zeros((len(batch_sizes), len(hidden_sizes)))
        loss_curves = {}
        for i, bs in enumerate(batch_sizes):
            for j, hs in enumerate(hidden_sizes):
                print(f"Entrenando: opt={optimizer}, lr={lr}, batch_size={bs}, hidden_size={hs}...")
                errors_total, loss_curve = train_autoencoder_return_loss(ACT_FUNC, optimizer, lr, bs, hs)

                # Guardar los resultados
                resultados.append({
                    "optimizer": optimizer,
                    "lr": lr,
                    "batch_size": bs,
                    "hidden_size": hs,
                    "error": errors_total
                })

                heatmap_errors[i, j] = errors_total
                loss_curves[(bs, hs)] = loss_curve
    
        print(resultados)
        
        plt.figure(figsize=(8, 6), dpi=120)
        plt.imshow(heatmap_errors, cmap="viridis")

        plt.colorbar(label="Errores totales")

        # Ejes correctos
        plt.xticks(range(len(hidden_sizes)), hidden_sizes)   # columnas
        plt.yticks(range(len(batch_sizes)), batch_sizes)     # filas

        plt.xlabel("Hidden neurons")
        plt.ylabel("Batch size")
        plt.title(f"Heatmap errores (opt={optimizer}, lr={lr})")
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
        
        

# Ordenar por error (de menor a mayor)
resultados_ordenados = sorted(resultados, key=lambda x: x["error"])

# Imprimir el ranking final
print("\n=== Ranking de configuraciones por error total ===")
for idx, r in enumerate(resultados_ordenados):
    print(f"{idx+1}. error={r['error']:.6f} | opt={r['optimizer']}, lr={r['lr']}, batch={r['batch_size']}, hidden={r['hidden_size']}")
