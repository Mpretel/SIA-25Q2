import numpy as np
import matplotlib.pyplot as plt
import random
from constants2 import *
from load_emojis import X, emoji_labels


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

# --- Añadir activación linear ---
def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

activation_functions = {
    'tanh': (tanh, tanh_derivative, (-1, 1), 0.0),
    'sigmoid': (sigmoid, sigmoid_derivative, (0, 1), 0.5),
    'softsign': (softsign, softsign_derivative, (-1, 1), 0.0),
    'linear': (linear, linear_derivative, (None, None), 0.0)
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
            d_output = -error
            #d_output = error * self.activation_deriv(self.final_input)
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
    
# --- Arquitectura VAE ---
# encoder outputs 2 * LATENT_DIM (mu and logvar concatenated)
encoder = MLP(n_input=32*32 + 1, n_hidden=HIDDEN, n_output=LATENT_DIM * 2,
              activation_function='sigmoid', learning_rate=LR, optimizer=OPTIMIZER)
# decoder takes latent dim + bias and outputs original dim (reconstruction)
decoder = MLP(n_input=LATENT_DIM + 1, n_hidden=HIDDEN, n_output=32*32,
              activation_function="sigmoid", learning_rate=LR, optimizer=OPTIMIZER)


data = (X + 1) / 2 # Escalados para usar sigmoid

# Entrenamiento VAE
best_loss = np.inf
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    idx = np.random.permutation(len(data))
    batches = [idx[i:i + BATCH_SIZE] for i in range(0, len(idx), BATCH_SIZE)]
    epoch_loss = 0.0

    for batch in batches:
        x = data[batch]  # (B, D)

        # bias en entrada del encoder
        x_bias = np.hstack([x, np.ones((x.shape[0], 1))])  # (B, D+1)

        # === Encoder forward: obtiene mu y logvar concatenados ===
        enc_out = encoder.forward(x_bias)  # (B, 2*LATENT_DIM)
        mu = enc_out[:, :LATENT_DIM]
        logvar = enc_out[:, LATENT_DIM:]

        # reparameterization
        eps = np.random.normal(size=mu.shape)
        sigma = np.exp(0.5 * logvar)
        z = mu + sigma * eps  # (B, LATENT_DIM)

        # añadir bias al z antes del decoder
        z_bias = np.hstack([z, np.ones((z.shape[0], 1))])  # (B, LATENT_DIM+1)

        # === Decoder forward ===
        recon = decoder.forward(z_bias)  # (B, D)

        # === Pérdidas por muestra ===
        # Reconstrucción (MSE)
        eps_num = 1e-9

        # Binary Cross Entropy por muestra
        bce_per_sample = np.mean(
            -x * np.log(recon + eps_num) - (1 - x) * np.log(1 - recon + eps_num),
            axis=1
        )

        # KL por muestra
        kl_per_sample = 0.5 * np.sum(
            mu**2 + np.exp(logvar) - logvar - 1,
            axis=1
        )

        batch_bce = np.mean(bce_per_sample)
        batch_kl  = np.mean(kl_per_sample)

        loss = batch_bce + batch_kl

        epoch_loss += loss * len(batch)

        # === Backprop ===
        # 1) backward decoder: obtenemos grad wrt z_bias (incluye grad del bias)
        decoder_grad_zbias = decoder.backward(z_bias, y=x, output=recon)  # (B, LATENT_DIM+1)
        decoder_grad_z = decoder_grad_zbias[:, :-1]  # (B, LATENT_DIM)

        # 2) construir gradientes para mu y logvar que entran al encoder:
        # Gradientes de la KL:
        # dKL/dmu = mu
        dKL_dmu = mu
        dKL_dlogvar = 0.5 * (np.exp(logvar) - 1)

        # Grad through reparameterization
        grad_mu_from_dec = decoder_grad_z
        grad_logvar_from_dec = decoder_grad_z * (0.5 * sigma * eps)

        grad_mu_total = grad_mu_from_dec + dKL_dmu
        grad_logvar_total = grad_logvar_from_dec + dKL_dlogvar

        grad_enc_out = np.hstack([grad_mu_total, grad_logvar_total])
        encoder.backward(x_bias, grad_output=grad_enc_out)

    # Fin batches
    epoch_loss /= len(data)

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Loss={epoch_loss:.6f} | KL={batch_kl:.6f}")

    # Early stopping (igual que antes)
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

# usar mejores pesos
encoder, decoder = best_encoder, best_decoder

# Obtener mu del encoder
data_bias = np.hstack([data, np.ones((data.shape[0], 1))])
enc_out = encoder.predict(data_bias)
mu = enc_out[:, :LATENT_DIM]

#Plot espacio latente
plt.figure(figsize=(6,6))
plt.scatter(mu[:,0], mu[:,1])
for i in range(len(emoji_labels)):
    plt.text(mu[i,0], mu[i,1], emoji_labels[i], fontsize=9)
plt.colorbar()
plt.title("Espacio Latente (mu)")
plt.xlabel("z1")
plt.ylabel("z2")
plt.grid()
plt.show()

N_SAMPLES = 20

# sampleo de z ~ N(0, I)

#z = mu+sigma*np.random.normal(N_SAMPLES, LATENT_DIM)

# z = np.random.normal(loc=mu, scale=sigma, size=(N_SAMPLES, LATENT_DIM))

# agregar bias


for j in range(N_SAMPLES):
    z1 = input("z1: ")
    z2 = input("z2: ")
    z = np.array([[float(z1), float(z2)]])
    z_bias = np.hstack([z, np.ones((1, 1))])

    samples = decoder.predict(z_bias)   # (N, D)

# si querés binarizar (por prob 0.5):
    samples_bin = (samples > 0.5).astype(np.float32)
    plt.imshow(samples_bin.reshape(32,32), cmap='gray')
    plt.title(f"Sample {j}")
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(mu[:,0], mu[:,1])
    for i in range(len(emoji_labels)):
        plt.text(mu[i,0], mu[i,1], emoji_labels[i], fontsize=9)
    plt.colorbar()
    plt.title("Espacio Latente (mu)")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.grid()
    plt.show()
    

# ============================
#  GRID VARIACIONAL 2D
# ============================

# #  Reconstrucciones de patrones
# def show_pattern(vec35, ax, title=None):
#     mat = vec35.reshape(32, 32)
#     ax.imshow(mat, cmap='gray_r', vmin=0, vmax=1)
#     ax.set_xticks([]); ax.set_yticks([])
#     if title:
#         ax.set_title(title, fontsize=9)

# zmin = z.min(axis=0) - 1
# zmax = z.max(axis=0) + 1

# nx, ny = 8, 8
# grid = []
# for i in range(ny):
#     for j in range(nx):
#         gx = zmin[0] + (zmax[0]-zmin[0])*j/(nx-1)
#         gy = zmin[1] + (zmax[1]-zmin[1])*i/(ny-1)
#         grid.append([gx,gy])
# grid = np.array(grid)

# grid_bias = np.hstack([grid, np.ones((grid.shape[0],1))])
# gen = decoder.predict(grid_bias)

# gen = (gen >= 0.5).astype(float)

# plt.figure(figsize=(6,6))
# for i in range(nx*ny):
#     ax = plt.subplot(ny,nx,i+1)
#     show_pattern(gen[i], ax)
# plt.suptitle("Grid del espacio latente (VAE)")
# plt.show()