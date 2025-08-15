import json
import sys
import numpy as np
import matplotlib.pyplot as plt


from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

factory = PokemonFactory("pokemon.json")

list_pokemons = ["caterpie", "jolteon", "mewtwo", "snorlax", "onix"]
list_pokeballs = ["ultraball"]
list_lvl = np.arange(1, 100, 2)

hp = 1 # 0 a 1
nit = 100
salud = StatusEffect.NONE

dpokemon = {}

for pokemon in list_pokemons:
    dpokemon[pokemon] = {}  # diccionario para las pokeballs
    for pokeball in list_pokeballs:
        dpokemon[pokemon][pokeball] = {}  # diccionario para estados de salud
        for lvl in list_lvl:
            pok_estado = factory.create(pokemon, lvl, salud, hp)
            resultados = []
            for _ in range(nit):
                booleano, proba = attempt_catch(pok_estado, pokeball)
                resultados.append(booleano)
            cantidad_true = sum(resultados)
            frec_true = cantidad_true / len(resultados)
            
            dpokemon[pokemon][pokeball][lvl] = (frec_true, proba)
            print(f"{pokemon} con {pokeball} y {lvl} -> {frec_true:.2%} Proba, {proba:.4f}")


import matplotlib.pyplot as plt
import numpy as np

pokemons = list_pokemons
pokeballs = list_pokeballs

colores_pokemon = {
    "caterpie": "green",
    "jolteon": "yellow",
    "mewtwo": "purple",
    "snorlax": "blue",
    "onix": "gray"
}

plt.figure(figsize=(10,6))

for pokemon in pokemons:
    for pokeball in pokeballs:
        frec_values = []
        for hp in list_lvl:
            frec_true, _ = dpokemon[pokemon][pokeball][hp]
            frec_values.append(frec_true)
        plt.plot(list_lvl, frec_values, marker='o', color=colores_pokemon[pokemon], label=f"{pokemon} - {pokeball}")

plt.xlabel("LVL")
plt.ylabel("Frecuencia relativa de captura")
#plt.title("Frecuencia de captura vs LVL para cada Pokémon y Pokéball")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()