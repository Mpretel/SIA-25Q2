import json
import argparse
from ga import GeneticAlgorithm


def main():
    parser = argparse.ArgumentParser(description="Algoritmo genético de triángulos")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON de configuración")
    args = parser.parse_args()

    # Cargar json
    with open(args.config, "r") as f:
        config = json.load(f)

    # Crear GA con los parámetros del json
    ga = GeneticAlgorithm(
        target_path=config["target_path"],
        canvas_size=tuple(config["canvas_size"]),
        n_triangles=config["n_triangles"],
        pop_size=config["pop_size"],
        generations=config["generations"],
        crossover_rate=config["crossover_rate"],
        mutation_rate=config["mutation_rate"],
        elitism=config["elitism"],
        tournament_k=config["tournament_k"],
        out_dir=config["out_dir"]
    )

    ga.run()

if __name__ == "__main__":
    main()