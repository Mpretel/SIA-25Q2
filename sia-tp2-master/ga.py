#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random, os, time
import cv2

from constants import MIN_RGB, MAX_RGB, MIN_ALPHA, MAX_ALPHA

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
            for i in range(n_triangles):
                # RGBA con límites específicos
                r = np.random.randint(MIN_RGB, MAX_RGB + 1)
                g = np.random.randint(MIN_RGB, MAX_RGB + 1)
                b = np.random.randint(MIN_RGB, MAX_RGB + 1)
                a = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)

                # 3 vértices (X,Y) ∈ [0, 255]
                coords = np.random.randint(0, 256, size=6)

                # Guardar en el gen
                self.genes[i] = np.array([r, g, b, a, *coords], dtype=np.uint8)

            #self.genes = np.random.randint(0, 256, size=(n_triangles, 10), dtype=np.uint8)
        
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

        # Dibujar triángulos en orden de alfa (de mayor a menor)
        sorted_genes = sorted(self.genes, key=lambda triangle: triangle[3], reverse=True)

        for triangle in sorted_genes:
            triangle = triangle.astype(np.int32)
            # Primeras 4 posiciones: RGBA
            r, g, b, a = [int(x) for x in triangle[:4]]
            # Siguientes 6 posiciones: 3x(X,Y) de los vértices
            xs = [int(triangle[4+i] * (W-1) / 255) for i in range(3)]
            ys = [int(triangle[7+i] * (H-1) / 255) for i in range(3)]
            poly = list(zip(xs, ys))
            # Dibujar triángulo en una capa aparte y combinar
            layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer, 'RGBA')
            draw.polygon(poly, fill=(r, g, b, a))
            base = Image.alpha_composite(base, layer)
        return base

    # Calcula el fitness como similarity (a maximizar)
    def evaluate_fitness(self, target_rgb):
        rendered = np.array(self.render().convert('RGB'), dtype=np.uint8)
        r = rendered.astype(np.int32)
        t = target_rgb.astype(np.int32)
        mae = np.mean(np.abs(r - t))
        #sim = 1/mae  # entre 0 y 1
        sim = 1.0 - (mae / 255.0)
        self.fitness = sim
        return sim

# ---------------------------
# Clase del Algoritmo Genético
# ---------------------------
class GeneticAlgorithm:
    def __init__(self, target_path, canvas_size, n_triangles, pop_size, kids_size,
                 parents_selection_method, crossover_method, mutation_method, gen_selection_method, end_criteria, out_dir, k_threshold=0.7):
        self.target_path = target_path
        self.canvas_size = canvas_size
        self.n_triangles = n_triangles
        self.pop_size = pop_size
        self.kids_size = kids_size 
        # kids_size debe ser par, sino se descarta el último
        if self.kids_size % 2 != 0:
            self.kids_size -= 1
            print(f"kids_size debe ser par, se ajusta a {self.kids_size}")
        self.parents_selection_method = parents_selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.gen_selection_method = gen_selection_method
        self.end_criteria = end_criteria
        self.out_dir = out_dir
        self.best_fit_history = []
        self.target_img = Image.open(target_path).convert('RGB').resize(canvas_size, Image.LANCZOS)
        self.target_rgb = np.array(self.target_img, dtype=np.uint8)

        self.population = [Individual(n_triangles, canvas_size) for _ in range(pop_size)]
        self.best = None
        self.best_fitness = -1
    
    def select(self, individuals, k, method):
        """
        Selects k individuals based on the specified selection method.
        Supports 'elitism', 'roulette', 'boltzmann', 'ranking', 'deterministic_tournament', and 'probabilistic_tournament' methods.
        - 'elitism' selects the top k individuals.
        - 'roulette' selects k individuals based on fitness proportionate selection.
        - 'deterministic_tournament'
        - 'probabilistic_tournament' 
        """
        # Elitismo
        if method["name"] == "elitism":
            # que pasa si k > pop_size?
            sorted_pop = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)
            return sorted_pop[:k]

        # Ruleta
        elif method["name"] == "roulette":
            total_fit = sum(individual.fitness for individual in individuals)
            r = random.uniform(0, total_fit)
            acum = 0
            for individual in individuals:
                acum += individual.fitness
                if acum >= r:
                    return individual
            return individuals[-1]

        # Torneo determinístico
        elif method["name"] == "deterministic_tournament":
            competitors = random.sample(individuals, self.tournament_k)
            return max(competitors, key=lambda individual: individual.fitness)

        # Torneo probabilístico
        elif method["name"] == "probabilistic_tournament":
            competitors = random.sample(individuals, 2)
            if random.random() < self.k_threshold:
                return max(competitors, key=lambda individual: individual.fitness)
            else:
                return min(competitors, key=lambda individual: individual.fitness)
        

    def crossover(self, parent_a, parent_b):
        """
        Performs crossover between two parents to produce two children.
        Allows for no crossover based on crossover rate: if no crossover, children are exact copies of parents.
        Supports 'uniform' and 'one-point' crossover methods.
        - uniform crossover exchanges genes based on a given rate.
        - one-point crossover exchanges genes from a single point and onwards.
        """
        # account for crossover rate
        if random.random() > self.crossover_rate:
            # No crossover, childs are exact copies of parents
            return (
                Individual(self.n_triangles, self.canvas_size, genes=parent_a.genes.copy()),
                Individual(self.n_triangles, self.canvas_size, genes=parent_b.genes.copy())
            )
        
        # One-point crossover: select a point and exchange all genes after that point
        if self.crossover_method["name"] == "one-point":
            point = np.random.randint(1, self.n_triangles)
            child1_genes = np.concatenate([parent_a.genes[:point], parent_b.genes[point:]])
            child2_genes = np.concatenate([parent_b.genes[:point], parent_a.genes[point:]])
            return (
                Individual(self.n_triangles, self.canvas_size, genes=child1_genes),
                Individual(self.n_triangles, self.canvas_size, genes=child2_genes)
            )
        
        # Uniform crossover: exchange genes based on given rate
        elif self.crossover_method["name"] == "uniform":
            mask = np.random.rand(self.n_triangles) < self.crossover_method["rate"]
            
            child1_genes = parent_a.genes.copy()
            child1_genes[mask] = parent_b.genes[mask]

            child2_genes = parent_b.genes.copy()
            child2_genes[mask] = parent_a.genes[mask]

            return (
                Individual(self.n_triangles, self.canvas_size, genes=child1_genes),
                Individual(self.n_triangles, self.canvas_size, genes=child2_genes)
            )

    def mutate(self, individual):
        """ 
        Mutates an individual based on the specified mutation method.
        Supports limited and uniform multigene mutation methods.
        - 'limited_multigene' selects M genes to mutate with a given rate --> if M > n_triangles, M = n_triangles - 1
        - 'uniform_multigene' replaces each gene with a probability defined by mutation rate.
        Mutation changes both color (RGBA) and position (XY) if mutation rate is applied.
        """
        mutation_rate = self.mutation_method["mutation_rate"]

        # Limited multi-gene mutation: selects M genes to mutate with given mutation rate
        if self.mutation_method["name"] == "limited_multigene":
            M = self.mutation_method["M"]
            if M > self.n_triangles:
                M = self.n_triangles - 1
            gene_indices = np.random.choice(self.n_triangles, M, replace=False)

            for idx in gene_indices:
                if random.random() < mutation_rate:
                    # Nuevo triángulo con restricciones
                    r = np.random.randint(MIN_RGB, MAX_RGB + 1)
                    g = np.random.randint(MIN_RGB, MAX_RGB + 1)
                    b = np.random.randint(MIN_RGB, MAX_RGB + 1)
                    a = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)
                    coords = np.random.randint(0, 256, size=6)
                    individual.genes[idx] = np.array([r, g, b, a, *coords], dtype=np.uint8)

                    #individual.genes[idx] = np.random.randint(0, 256, size=individual.genes[idx].shape, dtype=np.uint8)

        # Uniform multi-gene mutation: mutate each gene with a probability defined by mutation rate
        elif self.mutation_method["name"] == "uniform_multigene":
            for triangle in individual.genes:
                if random.random() < mutation_rate:
                    # R, G, B
                    for i in [0, 1, 2]:
                        triangle[i] = np.random.randint(MIN_RGB, MAX_RGB + 1)
                    # alpha
                    triangle[3] = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)
                    # coordenadas X, Y
                    for i in range(4, 10):
                        triangle[i] = np.random.randint(0, 256)

    # Graficar evolución del fitness
    def plot_fitness(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.best_fit_history)+1), self.best_fit_history, marker="o")
        plt.title("Evolución del fitness")
        plt.xlabel("Generación")
        plt.ylabel("Best fitness")
        plt.grid(True)
        plt.show()

    def evaluate_end_criteria(self, gen, start_time):
        if self.end_criteria["name"] == "generations":
            return gen >= self.end_criteria["value"]
        elif self.end_criteria["name"] == "fitness":
            return self.best_fitness >= self.end_criteria["value"]
        elif self.end_criteria["name"] == "time":
            elapsed = time.time() - start_time
            return elapsed >= self.end_criteria["value"]
        else:
            raise ValueError("Unknown end criteria")

    # Ejecuta el algoritmo genético
    def run(self):
        start_time = time.time()        
        gen = 0

        while not self.evaluate_end_criteria(gen, start_time):
            # Evaluar población
            for individual in self.population:
                if individual.fitness is None:
                    individual.evaluate_fitness(self.target_rgb)

            # Encontrar el mejor de la generación
            gen_best = max(self.population, key=lambda individual: individual.fitness)
            self.best_fit_history.append(gen_best.fitness)

            if gen_best.fitness > self.best_fitness:
                self.best = gen_best
                self.best_fitness = gen_best.fitness

                # Actualizar la figura en tiempo real
                img_rgb = np.array(gen_best.render().convert("RGB"))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Best Individual", img_bgr)
                if cv2.waitKey(1000) & 0xFF == 27:  # 100 ms, ESC para salir
                    break

                # Guardar la mejor imagen cada vez que se mejora
                self.best.render().save(os.path.join(self.out_dir, f'best_gen{gen:05d}.png'))

            if gen % 10 == 0 or gen == 0: # cada 10 generaciones se imprime el fitness
                elapsed = time.time() - start_time
                print(f"Gen {gen:4d} | best {gen_best.fitness:.2f} | overall {self.best_fitness:.2f} | {elapsed:.1f}s")

            # Se seleccionan los k padres a cruzar
            parents_to_breed = self.select(individuals=self.population, k=self.kids_size, method=self.parents_selection_method)

            # Nueva población
            kids = []

            # Elitismo
            #elites = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)[:self.elitism]
            #kids.extend([Individual(self.n_triangles, self.canvas_size, genes=e.genes.copy()) for e in elites])

            # Recorrer de a pares de padres (k/2 pares)
            for i in range(0, len(parents_to_breed), 2):
                parent_a = parents_to_breed[i]
                parent_b = parents_to_breed[i+1]

                # Cruzar 2 padres -> genera 2 hijos
                child1, child2 = self.crossover(parent_a, parent_b)

                # Mutación
                self.mutate(child1)
                self.mutate(child2)

                kids.append(child1)
                kids.append(child2)
                

            # Selección/ sesgo joven o tradicional según aptitud

            self.population = kids

            gen += 1

        cv2.destroyAllWindows()

        print("Finalizado. Mejor similarity:", np.round(self.best_fitness, 3))
        # print("Finalizado. Mejor similarity:", self.best_fitness, "->", self.best_fitness * 100.0, "%") normalizado
        self.plot_fitness()
        self.best.render().save("best_final.png")