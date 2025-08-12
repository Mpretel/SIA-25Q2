import json
import sys
import numpy as np
import matplotlib.pyplot as plt


from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

factory = PokemonFactory("pokemon.json")

list_pokemons = ["snorlax", "caterpie", "jolteon", "mewtwo", "onix"]
list_pokeballs = ["pokeball"]
list_hp = np.arange(0, 1.1, 0.1)  # 0 a 1 con paso de 0.1


lvl = 100 # 1 a 100
nit = 10000
salud = StatusEffect.NONE

dpokemon = {}

for pokemon in list_pokemons:
    dpokemon[pokemon] = {}  # diccionario para las pokeballs
    for pokeball in list_pokeballs:
        dpokemon[pokemon][pokeball] = {}  # diccionario para estados de salud
        for hp in list_hp:
            pok_estado = factory.create(pokemon, lvl, salud, hp)
            resultados = []
            for _ in range(nit):
                booleano, proba = attempt_catch(pok_estado, pokeball)
                resultados.append(booleano)
            cantidad_true = sum(resultados)
            frec_true = cantidad_true / len(resultados)
            
            dpokemon[pokemon][pokeball][hp] = (frec_true, proba)
            print(f"{pokemon} con {pokeball} y {hp*100} -> {frec_true:.2%} Proba, {proba:.4f}")


import matplotlib.pyplot as plt
import numpy as np

pokemons = list_pokemons
pokeballs = list_pokeballs

plt.figure(figsize=(10,6))

for pokemon in pokemons:
    for pokeball in pokeballs:
        frec_values = []
        for hp in list_hp:
            frec_true, _ = dpokemon[pokemon][pokeball][hp]
            frec_values.append(frec_true)
        plt.plot(list_hp, frec_values, marker='o', label=f"{pokemon} - {pokeball}")

plt.xlabel("HP (proporción)")
plt.ylabel("Frecuencia relativa de captura (frec_true)")
plt.title("Frecuencia de captura vs HP para cada Pokémon y Pokéball")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()