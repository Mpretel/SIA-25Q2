import json
import sys
import numpy as np
import matplotlib.pyplot as plt


from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

factory = PokemonFactory("pokemon.json")
with open(f"{sys.argv[1]}", "r") as f:
    config = json.load(f)
    ball = config["pokeball"]
    pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, 1)

list_pokemons = ["caterpie", "jolteon", "mewtwo", "snorlax", "onix"]
list_pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]

dpokemon = {}

lvl = 100 # 1 a 100
hp = 1 # 0 a 1
nit = 100

for pokemon in list_pokemons:
    dpokeball = {}
    dpokemon.setdefault(pokemon, dpokeball)
    pok_estado = factory.create(pokemon, lvl, StatusEffect.NONE, hp)
    for pokeball in list_pokeballs:
        resultados = []
        for _ in range(nit):
            bool, proba = attempt_catch(pok_estado, pokeball)
            resultados.append(bool)
        cantidad_true = sum(resultados)
        frec_true = cantidad_true / len(resultados)
        
        dpokemon[pokemon, pokeball] = (frec_true, proba)
        print(f"{pokemon} con {pokeball} -> {frec_true:.2%} Proba, {proba:.4f}")


# Colores por Pokémon
colores_pokemon = {
    "caterpie": "green",
    "jolteon": "yellow",
    "mewtwo": "purple",
    "snorlax": "blue",
    "onix": "gray"
}

# Texturas por pokeball
texturas_pokeball = {
    "pokeball": "/",
    "ultraball": "\\",
    "fastball": "|",
    "heavyball": "-"
}

labels = []
frecuencias = []
colores = []
texturas = []

for pokemon in list_pokemons:
    for pokeball in list_pokeballs:
        key = (pokemon, pokeball)
        frec_true, _ = dpokemon[key]
        labels.append(f"{pokemon}\n{pokeball}")
        frecuencias.append(frec_true)
        colores.append(colores_pokemon[pokemon])
        texturas.append(texturas_pokeball[pokeball])

x = np.arange(len(labels))

plt.figure(figsize=(12,6))

bars = []
for i in range(len(labels)):
    bar = plt.bar(x[i], frecuencias[i], color=colores[i], hatch=texturas[i])
    bars.append(bar)

plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel("Frecuencia relativa de captura (frec_true)")
plt.title("Frecuencia de captura para cada Pokémon y tipo de Pokéball")

for bar, freq in zip(bars, frecuencias):
    # bar es un container, extraigo la barra
    rect = bar.patches[0]
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height + 0.01, f"{freq:.2}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

#################3

# Los pokemons y pokeballs que usás
pokemons = list_pokemons
pokeballs = list_pokeballs

# Armo una matriz: filas pokemons, columnas pokeballs normalizadas por pokeball básica
norm_frec = []

for pokemon in pokemons:
    base = dpokemon[(pokemon, "pokeball")][0]  # frecuencia con pokeball básica
    fila = []
    for pb in pokeballs:
        frec = dpokemon[(pokemon, pb)][0]
        norm = frec / base if base != 0 else 0  # evitar división por 0
        fila.append(norm)
    norm_frec.append(fila)

norm_frec = np.array(norm_frec)  # shape (5 pokemons, 4 pokeballs)

# Ahora graficamos barras agrupadas por pokeball
x = np.arange(len(pokeballs))  # posiciones para cada tipo de pokeball

width = 0.15  # ancho de cada barra
fig, ax = plt.subplots(figsize=(10,6))

colores_pokemon = {
    "caterpie": "green",
    "jolteon": "yellow",
    "mewtwo": "purple",
    "snorlax": "blue",
    "onix": "gray"
}

for i, pokemon in enumerate(pokemons):
    ax.bar(x + i*width, norm_frec[i], width, label=pokemon, color=colores_pokemon[pokemon])

# Etiquetas y leyenda
ax.set_xticks(x + width*2)  # centrar los grupos en el eje x
ax.set_xticklabels(pokeballs)
ax.set_ylabel("Frecuencia relativa normalizada\nrespecto a Pokeball básica")
ax.set_title("Comparación de efectividad de Pokéballs normalizada por Pokémon")
ax.axhline(1, color='black', linestyle='--', linewidth=0.7)  # línea base en 1

ax.legend(title="Pokémon")
plt.tight_layout()
plt.show()
