import numpy as np
import matplotlib.pyplot as plt

# Rango de valores de mae entre 0 y 1
mae = np.linspace(0, 255, 500)

# Función de fitness
fitness = 1 - (mae / 255)

# Graficar
plt.plot(mae, fitness, label=r"$1 - \frac{MAE}{255}$", linewidth=2.5)
plt.xlabel("MAE", fontsize=16)
plt.ylabel("Fitness", fontsize=16)
#plt.title("Función de Fitness")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.show()