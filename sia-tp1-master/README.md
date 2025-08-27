# Sokoban Solver & Player

Este proyecto implementa el clásico juego **Sokoban**, con dos modos de uso:  
- *Play*: jugar manualmente usando el teclado.  
- *Solve*: resolver automáticamente los niveles usando distintos algoritmos de búsqueda (BFS, DFS, Greedy, A*).  

---

## Estructura del proyecto

    .
    ├──sokoban.py # Código
    └──levels/ # Carpeta con los distintos niveles del juego en formato .txt
        ├── 1.txt
        ├── 2.txt
        └── ...

### Niveles

Cada archivo dentro de `levels/` representa un nivel del juego, usando caracteres ASCII:  

- `#` → Pared  
- `@` → Jugador  
- `$` → Caja  
- `.` → Objetivo  
- `*` → Caja sobre objetivo  
- `+` → Jugador sobre objetivo  
- ` ` → Espacio vacío  

#### Agregar nuevos niveles

Para agregar un nuevo nivel, creá un archivo `.txt` en la carpeta `levels/` con el diseño del nivel usando los caracteres mencionados. Nuevos niveles pueden ser descargados desde [Sokoban Levels](http://game-sokoban.com/index.php?mode=catalog). 

---

## Ejecución

### Requisitos

- Python 3.8+  
- No requiere librerías externas (solo módulos estándar de Python).

### Instrucciones

Ejecutá el código:

   ```bash
   $ python sokoban.py
   ```

y seguí las instrucciones en pantalla:

1. Elegir un nivel:  El programa mostrará todos los niveles disponibles dentro de la carpeta `levels/`. Escribí el nombre del nivel que quieras jugar y presioná `Enter`.

2. Elegir un modo de juego: El programa preguntará si querés jugar manualmente (`play`) o resolver automáticamente el nivel con un algoritmo de búsqueda (`solve`). Ingresá uno de los siguientes modos y presioná `Enter`:
- `play` → Modo manual: se mostrará el tablero en la terminal y podrás empezar a mover al jugador usando el teclado:
    - W → Mover hacia arriba (↑)
    - A → Mover hacia la izquierda (←)
    - S → Mover hacia abajo (↓)
    - D → Mover hacia la derecha (→)

    El objetivo es empujar todas las cajas ($) sobre los objetivos (.).
    Si el jugador queda en un deadlock (una caja bloqueada en una esquina), el juego termina y se muestra un mensaje indicando que se perdió.

    Al completar el nivel, se mostrará la cantidad de movimientos realizados y se dará la opción de repetir la animación de la solución si se presiona .

- `solve` → Modo automático: el programa resolverá el nivel usando un algoritmo de búsqueda y mostrará la solución.
Se te pedirá que elijas uno de los siguientes algoritmos:
    - Métodos de búsqueda desinformada:
        - `bfs` → Búsqueda en amplitud (Breadth-First Search)
        - `dfs` → Búsqueda en profundidad (Depth-First Search)
    - Métodos de búsqueda informada:
        - `greedy` → Búsqueda voraz (Global Greedy Search)
        - `a_star` → Búsqueda A estrella (A*)

        Luego, si elegiste `greedy` o `a_star`, se te pedirá que elijas una heurística:

        - `misplaced` → Cantidad de cajas fuera de lugar (no sobre un objetivo)
        - `manhattan` → Mínima distancia Manhattan total de las cajas a los objetivos

    Al finalizar la búsqueda, el programa mostrará la solución encontrada, indicando la cantidad de movimientos de la solución, la cantidad de nodos expandidos y el tiempo tomado para encontrar la solución. Nuevamente, se dará la opción de repetir la animación de la solución si se presiona `Enter`.


#### Ejemplo de ejecución:

```bash$ python sokoban.py
Choose a level (1/2/3): 1
Choose a mode (play/solve): play
```

o

```bash$ python sokoban.py
Choose a level (1/2/3): 1
Choose a mode (play/solve): solve
Choose a method (bfs/dfs/greedy/a_star): a_star
Choose a heuristic (misplaced/manhattan): manhattan
```


---

## Presentación 

En el siguiente enlace se encuentra la presentación del proyecto: [Presentación](https://docs.google.com/presentation/d/1sGi8b1btgXBkf8TyNUAc4pHzL40Jp0Am9cf3RH-re6o/edit?usp=sharing)
