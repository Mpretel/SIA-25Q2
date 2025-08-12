import json
import sys
import numpy as np
import matplotlib.pyplot as plt


from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

factory = PokemonFactory("pokemon.json")

list_pokemons = ["caterpie"]
list_pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]
list_salud = [StatusEffect.NONE, StatusEffect.BURN, StatusEffect.POISON, StatusEffect.FREEZE, StatusEffect.SLEEP, StatusEffect.PARALYSIS]
list_salud_names = ["none", "burn", "poison", "freeze", "sleep", "paralysis"]


lvl = 100 # 1 a 100
hp = 0.00000000000001 # 0 a 1
nit = 100

dpokemon = {}

for pokemon in list_pokemons:
    dpokemon[pokemon] = {}  # diccionario para las pokeballs
    for pokeball in list_pokeballs:
        dpokemon[pokemon][pokeball] = {}  # diccionario para estados de salud
        for salud, salud_name in zip(list_salud, list_salud_names):
            pok_estado = factory.create(pokemon, lvl, salud, hp)
            resultados = []
            for _ in range(nit):
                booleano, proba = attempt_catch(pok_estado, pokeball)
                resultados.append(booleano)
            cantidad_true = sum(resultados)
            frec_true = cantidad_true / len(resultados)
            
            dpokemon[pokemon][pokeball][salud_name] = (frec_true, proba)
            print(f"{pokemon} con {pokeball} y {salud} -> {frec_true:.2%} Proba, {proba:.4f}")


import matplotlib.pyplot as plt
import numpy as np

pokemon = "caterpie"  # fijamos Pokémon, por ejemplo
pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]

x = np.arange(len(pokeballs))  # posición de cada tipo de pokeball
width = 0.13  # ancho de barra

fig, ax = plt.subplots(figsize=(12,6))

# Colores para cada estado de salud (puedes elegir otros)
colores_salud = {
    "none": "gray",
    "poison": "purple",
    "burn": "red",
    "paralysis": "yellow",
    "sleep": "blue",
    "freeze": "cyan"
}

for i, salud in enumerate(list_salud_names):
    valores = []
    for pokeball in pokeballs:
        frec_true, _ = dpokemon[pokemon][pokeball][salud]
        valores.append(frec_true)
    # Posición desplazada para cada grupo de barras (estado de salud)
    pos = x + i*width
    ax.bar(pos, valores, width, label=salud, color=colores_salud[salud])

ax.set_xticks(x + width*(len(list_salud_names)/2 - 0.5))
ax.set_xticklabels(pokeballs)
ax.set_ylabel("Frecuencia relativa de captura")
ax.set_title(f"Frecuencia de captura de {pokemon} por tipo de Pokéball y estado de salud")
ax.legend(title="Estado de salud")
plt.tight_layout()
plt.show()
