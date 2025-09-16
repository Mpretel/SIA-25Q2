import os
import json
import argparse
from ga import GeneticAlgorithm
from PIL import Image
import shutil

from constants import CONFIGS_DIR, INPUT_IMAGES_DIR, OUTPUT_IMAGES_DIR, SCALE_FACTOR, RGB

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Algoritmo genético de triángulos")
    parser.add_argument("--config", type=str, required=True, help="Nombre del archivo JSON de configuración")
    args = parser.parse_args()

    config_name = args.config

    # Add ".json" extension if the user didn’t include it
    if not config_name.endswith(".json"):
        config_name += ".json"

    # Validate that the configuration file exists inside the "configs" folder
    config_path = os.path.join(CONFIGS_DIR, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"\nNo se encontró el archivo de configuración {config_path}")

    # Load the configuration JSON
    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate that the input images folder exists
    if not os.path.exists(INPUT_IMAGES_DIR):
        raise FileNotFoundError(f"\nNo se encontró la carpeta {INPUT_IMAGES_DIR}")

    # Get all image files in input_images (PNG, JPG, JPEG)
    images = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not images:
        raise FileNotFoundError(f"\nNo hay imágenes en {INPUT_IMAGES_DIR}")

    # Show available images to the user
    print("\nImágenes disponibles:")
    for i, img in enumerate(images):
        print(f"[{i}] {img}")

    # Ask the user to choose one image
    choice = -1
    while choice < 0 or choice >= len(images):
        choice = int(input("\nElige el número de la imagen a usar: "))

    target_path = os.path.join(INPUT_IMAGES_DIR, images[choice])

    # Ask the user for the number of triangles
    n_triangles = 0
    while n_triangles <= 0 or not isinstance(n_triangles, int):
        n_triangles = int(input("\nIngrese la cantidad de triángulos: "))

    # Get target image size (scaled down by SCALE_FACTOR)
    with Image.open(target_path) as img:
        w, h = img.size
        canvas_size = (w // SCALE_FACTOR, h // SCALE_FACTOR)

    # Create output folder if it doesn’t exist
    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)
    
    # Create a specific subfolder for this run (based on image name + n_triangles + color mode + config name)
    image_filename = os.path.splitext(images[choice])[0]
    if RGB:
        color_mode = "RGB"
    else:
        color_mode = "HSV"
    run_output_dir = f"{image_filename}_{n_triangles}_{color_mode}_config_{os.path.splitext(os.path.basename(args.config))[0]}"
    run_output_path = os.path.join(OUTPUT_IMAGES_DIR, run_output_dir)
    if not os.path.exists(run_output_path):
        os.makedirs(run_output_path)

    # Save a copy of the config file into the run’s output folder
    shutil.copy(config_path, os.path.join(run_output_path, "config.json"))

    # Initialize the Genetic Algorithm with hyperparameters from the config
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

    # Run the Genetic Algorithm
    ga.run()


if __name__ == "__main__":
    main()
