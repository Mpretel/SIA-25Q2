# Algoritmos Genéticos para Aproximación de Imágenes con Triángulos

Este proyecto implementa un compresor de imágenes que aproxima una imagen de entrada mediante una composición de triángulos empleando algoritmos genéticos, evolucionando una población inicial de imágenes candidatas hasta obtener la mejor aproximación visual a la imagen original.

---

## 🚀 Estructura del Proyecto

- **`ga.py`**: contiene las clases principales
  - `Individual`: representa una solución (conjunto de triángulos).
  - `GeneticAlgorithm`: gestiona la evolución de la población inicial.
- **`main.py`**: script de entrada. Permite ejecutar el programa desde la consola a partir de un archivo de configuración, indicando la imagen de entrada y el número de triángulos a utilizar.
- **`constants.py`**: parámetros fijos de directorios y restricciones geométricas.
- **`configs/`**: carpeta de configuraciones en formato JSON.
  - **`default.json`**: archivo de configuración base para el algoritmo.

---

## ⚙️ Configuración

### `constants.py`
Contiene parámetros globales del programa:

```python
CONFIGS_DIR = "configs"          # Carpeta de configuraciones
INPUT_IMAGES_DIR = "input_images" # Carpeta con imágenes de entrada
OUTPUT_IMAGES_DIR = "ga_output"   # Carpeta de salida

SCALE_FACTOR = 2 # Factor de escalado de la imagen

MIN_ALPHA = 50 # Valor mínimo para el alfa (opacidad)
MAX_ALPHA = 255 # Valor máximo

MIN_AREA = 100 # Valor mínimo para el área
MAX_AREA = 1000000 # Valor máximo
SEED = 43 # Valor numérico para la semilal (o None)
RGB = True # True para RGB, False para HSV
DELTA = None # Valor numérico de Delta para mutación (o None)
```

### `default.json`
Contiene la configuración de la estructura propia del algoritmo genético:
```json
{
  "pop_size": 30, # N
  "kids_size": 10, # K
  "parents_selection_method": {
    "name": "elitism", # elitism, roulette, universal, ranking, boltzmann, determinist_tournament, probabilistic_tournament
    "Tc": 100, # Tc (boltzmann)
    "T0": 120, # T0 (boltzmann)
    "decay": 1, # decay (boltzmann)
    "M": 5, # Cantidad de individuos M de N (deterministic_tournament)
    "p": 0.5 threshold (probabilistic_tournament)
  },
  "crossover_criteria": {
    "method": {
      "name": "uniform", # one-point, uniform
      "p": 0.5 # Probabilidad de crossover (uniform)
    },
    "crossover_rate": 0.5 # Probabilidad de recombinación
  },
  "mutation_method": {
    "name": "uniform_multigene", # limited_multigene, uniform_multigene, complete
    "mutation_rate": 0.2, # Tasa de mutación (uniform_multigene)
    "M": 1 # Cantidad de genes a mutar (limited_multigene)
  },
  "new_gen_creation_criteria": "traditional", # traditional, young_bias
  "new_gen_selection_method": {
    "name": "elitism", # elitism, roulette, universal, ranking, boltzmann, determinist_tournament, probabilistic_tournament
    "Tc": 100, # Tc (boltzmann)
    "T0": 120, # T0 (boltzmann)
    "decay": 1, # decay (boltzmann)
    "M": 5, # Cantidad de individuos M de N (deterministic_tournament)
    "p": 0.5 threshold (probabilistic_tournament)
  },
  "end_criteria": {
    "name": "generations", # generations, fitness, time
    "value": 30000 # Cantidad de generaciones (generations), mejor fitness (fitness) ó tiempo (time)
  }
}
```
## 🎮 Ejecución

### `main.py`
Ejecuta el algoritmo genético a partir de un archivo de configuración:
```bash
python main.py --config default.json
```
Luego de ejecutar se deberá elegir el número de imagen (de la carpeta input_images) desde la consola:
```bash
Imágenes disponibles:
[0] caballero.jpg
[1] catan.jpg
[2] ghana.png
[3] ghana2.png
[4] noche.jpg
[5] payaso.jpg
[6] triangulo.png
[7] triangulos.png

Elige el número de la imagen a usar: 4
```
y el número de triángulos con los cuales comprimir la imagen:
```bash
Ingrese la cantidad de triángulos: 20
```
Al comenzar la compresión, en consola se irán mostrando el fitness cada 10 generaciones, junto con el mejor fitness histórico y el tiempo de procesado:
```bash
Gen    0 | best 0.66 | overall 0.66 | 0.5s
Gen   10 | best 0.70 | overall 0.70 | 2.3s
```
Al finalizar el programa (de acuerdo al criterio de corte configurado), la consola mostrará el mejor fitness alcanzado:
```bash
Finalizado. Mejor similarity: 0.857
```
y se guardará en el directorio de salida (ga_output) una carpeta con la mejor imagen generada (best_final.png) junto con la información de todos los triángulos que la componen (best_triangles.txt) y la evolución del fitness (fitness_evolution.png).
