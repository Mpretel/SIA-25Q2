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

SCALE_FACTOR = 2

MIN_ALPHA = 50
MAX_ALPHA = 255

MIN_AREA = 100
MAX_AREA = 1000000

SEED = 43
RGB = False
DELTA = 10
