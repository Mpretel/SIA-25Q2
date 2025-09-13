import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

csv_files = glob.glob("triangulos_fitness/*.csv")

plt.figure(figsize=(10, 6))

for file in csv_files:
    df = pd.read_csv(file)
    plt.plot(df["generation"], df["best_fitness"], label=os.path.splitext(os.path.basename(file))[0], linewidth=2)

plt.xlabel("Generación", fontsize=14)
plt.ylabel("Fitness", fontsize=14)
#plt.title("Evolución del Fitness", fontsize=16)
plt.legend(fontsize=10, loc='lower right')
plt.grid(True)
plt.show()
