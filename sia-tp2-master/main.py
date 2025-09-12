import os
import json
import argparse
from ga import GeneticAlgorithm
from PIL import Image
import shutil

from constants import CONFIGS_DIR, INPUT_IMAGES_DIR, OUTPUT_IMAGES_DIR, SCALE_FACTOR, RGB

def main():

    parser = argparse.ArgumentParser(description="Algoritmo genético de triángulos")
    parser.add_argument("--config", type=str, required=True, help="Nombre del archivo JSON de configuración")
    args = parser.parse_args()

    config_name = args.config

    # agregar el .json a la ruta si no lo tiene
    if not config_name.endswith(".json"):
        config_name += ".json"

    # Validar que el archivo de configuración existe en la carpeta llamada configs
    config_path = os.path.join(CONFIGS_DIR, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"\nNo se encontró el archivo de configuración {config_path}")

    # Cargar json
    with open(config_path, "r") as f:
        config = json.load(f)

    # Buscar imágenes en carpeta input_images
    if not os.path.exists(INPUT_IMAGES_DIR):
        raise FileNotFoundError(f"\nNo se encontró la carpeta {INPUT_IMAGES_DIR}")

    images = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not images:
        raise FileNotFoundError(f"\nNo hay imágenes en {INPUT_IMAGES_DIR}")

    # Mostrar lista de imágenes
    print("\nImágenes disponibles:")
    for i, img in enumerate(images):
        print(f"[{i}] {img}")

    # Select an image
    choice = -1
    while choice < 0 or choice >= len(images):
        choice = int(input("\nElige el número de la imagen a usar: "))

    target_path = os.path.join(INPUT_IMAGES_DIR, images[choice])

    # Preguntar cantidad de triángulos
    n_triangles = 0
    while n_triangles <= 0 or not isinstance(n_triangles, int):
        n_triangles = int(input("\nIngrese la cantidad de triángulos: "))

    # Obtener tamaño de la imagen target
    with Image.open(target_path) as img:
        w, h = img.size
        canvas_size = (w // SCALE_FACTOR, h // SCALE_FACTOR)

    # Crear carpeta de salida si no existe
    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)
    
    # Crear una carpeta dentro de OUTPUT_IMAGES_DIR para esta corrida con el nombre de la imagen y n_triángulos
    image_filename = os.path.splitext(images[choice])[0]
    if RGB:
        color_mode = "RGB"
    else:
        color_mode = "HSV"
    run_output_dir = f"{image_filename}_{n_triangles}_{color_mode}_config_{os.path.splitext(os.path.basename(args.config))[0]}"
    run_output_path = os.path.join(OUTPUT_IMAGES_DIR, run_output_dir)
    if not os.path.exists(run_output_path):
        os.makedirs(run_output_path)

    # Guarda copia del archivo de config en la carpeta de outputs
    shutil.copy(config_path, os.path.join(run_output_path, "config.json"))

    # Crear GA con los hiperparámetros del config
    ga = GeneticAlgorithm(
        target_path=target_path,
        canvas_size=canvas_size,
        n_triangles=n_triangles,
        pop_size=config["pop_size"],
        kids_size=config["kids_size"],
        parents_selection_method=config["parents_selection_method"],
        crossover_criteria=config["crossover_criteria"],
        mutation_method=config["mutation_method"],
        new_gen_creation_criteria=config["new_gen_creation_criteria"],
        new_gen_selection_method=config["new_gen_selection_method"],
        end_criteria=config["end_criteria"],
        out_dir=run_output_path
    )

    ga.run()


if __name__ == "__main__":
    main()
