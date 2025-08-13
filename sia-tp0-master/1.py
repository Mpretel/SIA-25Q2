import json
import sys
import numpy as np
import matplotlib.pyplot as plt


from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

factory = PokemonFactory("pokemon.json")

list_pokemons = ["caterpie", "jolteon", "mewtwo", "snorlax", "onix"]
list_pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]

dpokemon = {}

lvl = 100 # 1 a 100
hp = 1 # 0 a 1
nit = 10000

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


# Colores por pokeball
colores_pokeball = {
    "pokeball": "red",
    "ultraball": "gold",
    "fastball": "orange",
    "heavyball": "black"
}

n_pokemons = len(list_pokemons)
n_pokeballs = len(list_pokeballs)

# Posiciones base para cada Pokémon
x = np.arange(n_pokemons)
width = 0.18  # ancho de cada barra

plt.figure(figsize=(12,6))

# Dibujar barras agrupadas
for i, pokeball in enumerate(list_pokeballs):
    frecs = [dpokemon[(pokemon, pokeball)][0] for pokemon in list_pokemons]
    plt.bar(x + i*width - (width*(n_pokeballs-1)/2),
            frecs,
            width=width,
            color=colores_pokeball[pokeball],
            label=pokeball)

# Etiquetas y títulos
plt.xticks(x, list_pokemons)
plt.ylabel("Frecuencia relativa de captura")
#plt.title("Frecuencia de captura por Pokémon y tipo de Pokéball")
plt.legend(title="Pokéball")

# Etiquetas encima de las barras
for i, pokeball in enumerate(list_pokeballs):
    frecs = [dpokemon[(pokemon, pokeball)][0] for pokemon in list_pokemons]
    for j, freq in enumerate(frecs):
        xpos = j + i*width - (width*(n_pokeballs-1)/2)
        plt.text(xpos, freq + 0.01, f"{freq:.2}", 
                 ha='center', va='bottom', fontsize=8)

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
ax.set_ylabel("Frecuencia relativa normalizada\nrespecto a la pokebola básica")
#ax.set_title("Comparación de efectividad de Pokéballs normalizada por Pokémon")
ax.axhline(1, color='black', linestyle='--', linewidth=0.7)  # línea base en 1

ax.legend(title="Pokémon")
plt.tight_layout()
plt.show()
