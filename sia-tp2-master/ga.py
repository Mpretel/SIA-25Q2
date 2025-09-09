#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random, os, time

# ---------------------------
# Clase que representa un individuo
# ---------------------------
class Individual:
    # Inicializa la clase con n_triángulos y tamaño del canvas
    # Cada triángulo tiene 10 genes: RGBA + 3x(X,Y)
    def __init__(self, n_triangles, canvas_size, genes=None):
        self.n_triangles = n_triangles
        self.canvas_size = canvas_size
        # Los genes se inicializan aleatoriamente...
        if genes is None:
            self.genes = np.random.randint(0, 256, size=(n_triangles, 10), dtype=np.uint8)
        # ...o vienen de una generación previa
        else:
            self.genes = genes.astype(np.uint8)
        # El fitness comienza como None
        self.fitness = None

    # Renderiza la imagen del individuo
    def render(self):
        W, H = self.canvas_size
        # Canvas blanco
        base = Image.new('RGBA', (W, H), (255, 255, 255, 255))
        for tri in self.genes:
            tri = tri.astype(np.int32)
            # Primeras 4 posiciones: RGBA
            r, g, b, a = [int(x) for x in tri[:4]]
            # Siguientes 6 posiciones: 3x(X,Y) de los vértices
            xs = [int(tri[4+i] * (W-1) / 255) for i in range(3)]
            ys = [int(tri[7+i] * (H-1) / 255) for i in range(3)]
            poly = list(zip(xs, ys))
            # Dibujar triángulo en una capa aparte y combinar
            layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer, 'RGBA')
            draw.polygon(poly, fill=(r, g, b, a))
            base = Image.alpha_composite(base, layer)
        return base

    # Calcula el fitness como MSE (a minimizar).
    # def evaluate(self, target_rgb):
    #     rendered = np.array(self.render().convert('RGB'), dtype=np.uint8)
    #     diff = rendered.astype(np.int32) - target_rgb.astype(np.int32)
    #     mse = np.mean(diff * diff)
    #     self.fitness = mse
    #     return mse

    # Calcula el fitness como similarity (a maximizar)
    def evaluate(self, target_rgb):
        rendered = np.array(self.render().convert('RGB'), dtype=np.uint8)
        r = rendered.astype(np.int32)
        t = target_rgb.astype(np.int32)
        mae = np.mean(np.abs(r - t))
        sim = 1/mae  # entre 0 y 1
        # sim = 1.0 - (mae / 255.0) normalizado
        self.fitness = sim
        return sim

# ---------------------------
# Clase del Algoritmo Genético
# ---------------------------
class GeneticAlgorithm:
    def __init__(self, target_path, canvas_size, n_triangles, pop_size,
                 generations, crossover_rate, mutation_rate, elitism, tournament_k, out_dir):
        self.target_path = target_path
        self.canvas_size = canvas_size
        self.n_triangles = n_triangles
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.out_dir = out_dir

        os.makedirs(out_dir, exist_ok=True)
        self.target_img = Image.open(target_path).convert('RGB').resize(canvas_size, Image.LANCZOS)
        self.target_rgb = np.array(self.target_img, dtype=np.uint8)

        self.population = [Individual(n_triangles, canvas_size) for _ in range(pop_size)]
        self.best = None
        self.best_fitness = -1

    def tournament_selection(self):
        competitors = random.sample(self.population, self.tournament_k)
        return max(competitors, key=lambda ind: ind.fitness)

    # Crossover uniforme (P=0.5), cada alelo es un triángulo completo
    def crossover(self, parent_a, parent_b):
        if random.random() > self.crossover_rate:
            return Individual(self.n_triangles, self.canvas_size, genes=parent_a.genes.copy())
        mask = np.random.rand(self.n_triangles) < 0.5
        child_genes = parent_a.genes.copy()
        child_genes[mask] = parent_b.genes[mask]
        return Individual(self.n_triangles, self.canvas_size, genes=child_genes)

    # Reemplaza con una proba mutation_rate cada gen (RGBA o X/Y)
    def mutate(self, individual):
        flat = individual.genes.reshape(-1)
        mask = np.random.rand(flat.size) < self.mutation_rate
        flat[mask] = np.random.randint(0, 256, size=mask.sum(), dtype=np.uint8)
        individual.genes = flat.reshape(individual.genes.shape)

    def run(self):
        start_time = time.time()
        for gen in range(1, self.generations + 1):
            # Evaluar población
            for ind in self.population:
                if ind.fitness is None:
                    ind.evaluate(self.target_rgb)

            # Encontrar el mejor de la generación
            gen_best = max(self.population, key=lambda ind: ind.fitness)

            # Mostrar progreso en tiempo real
            # plt.ion()
            # fig, ax = plt.subplots()
            # im = ax.imshow(self.population[0].render())  # primer render
            # ax.set_title("Best individual")
            # plt.show()

            if gen_best.fitness > self.best_fitness:
                self.best = gen_best
                self.best_fitness = gen_best.fitness

                # # Actualizar la figura en tiempo real
                # im.set_data(self.best.render())
                # ax.set_title(f"Gen {gen} | fitness {self.best_fitness:.4f}")
                # plt.pause(0.001)

                # Opcional: guardar la mejor imagen cada vez que se mejora
                # self.best.render().save(os.path.join(self.out_dir, f'best_gen{gen:05d}.png'))
            # plt.ioff()
            # plt.show()

            if gen % 10 == 0 or gen == 1:
                elapsed = time.time() - start_time
                print(f"Gen {gen:4d} | best {gen_best.fitness:.2f} | overall {self.best_fitness:.2f} | {elapsed:.1f}s")

            # Nueva población
            new_pop = []
            # Elitismo
            elites = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:self.elitism]
            new_pop.extend([Individual(self.n_triangles, self.canvas_size, genes=e.genes.copy()) for e in elites])

            while len(new_pop) < self.pop_size:
                parent_a = self.tournament_selection()
                parent_b = self.tournament_selection()
                child = self.crossover(parent_a, parent_b)
                self.mutate(child)
                new_pop.append(child)

            self.population = new_pop

        print("Finalizado. Mejor similarity:", np.round(self.best_fitness, 3))
        # print("Finalizado. Mejor similarity:", self.best_fitness, "->", self.best_fitness * 100.0, "%") normalizado
        self.best.render().save("best_final.png")


# Correr desde consola

# if __name__ == "__main__":
#     target = input("Ruta a la imagen objetivo: ").strip()
#     n_tri = int(input("Cantidad de triángulos: "))
#     pop = int(input("Tamaño de la población: "))
#     gens = int(input("Cantidad de generaciones: "))
#     w = int(input("Ancho del canvas: "))
#     h = int(input("Alto del canvas: "))
#     mut = float(input("Tasa de mutación (ej. 0.02): "))

#     ga = GeneticAlgorithm(
#         target_path=target,
#         canvas_size=(w, h),
#         n_triangles=n_tri,
#         pop_size=pop,
#         generations=gens,
#         crossover_rate=0.7,
#         mutation_rate=mut,
#         elitism=1,
#         tournament_k=3,
#         out_dir="ga_output"
#     )
#     ga.run()

