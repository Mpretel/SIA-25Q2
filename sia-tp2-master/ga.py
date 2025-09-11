#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random, os, time
import cv2

from constants import MIN_ALPHA, MAX_ALPHA, MIN_AREA, SEED

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    
def triangle_area(coords):
    # coords = [x1,y1, x2,y2, x3,y3]
    x1, y1, x2, y2, x3, y3 = map(float, coords) 
    return abs(0.5 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))

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
            # Inicializar un array vacío para todos los triángulos
            self.genes = np.zeros((n_triangles, 10), dtype=np.uint8)

            for i in range(n_triangles):
                # RGBA con límites específicos
                r, g, b = np.random.randint(0, 256, size=3)
                a = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)

                # 3 vértices (X,Y) ∈ [0, 255], con área mínima
                if MIN_AREA:
                    while True:
                        coords = np.random.randint(0, 256, size=6)
                        if triangle_area(coords) >= MIN_AREA:
                            break

                # 3 vértices (X,Y) ∈ [0, 255]
                else:
                    coords = np.random.randint(0, 256, size=6)

                # Guardar en el gen i
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
            xs = [int(triangle[4] * (W-1) / 255),
                int(triangle[6] * (W-1) / 255),
                int(triangle[8] * (W-1) / 255)]

            ys = [int(triangle[5] * (H-1) / 255),
                int(triangle[7] * (H-1) / 255),
                int(triangle[9] * (H-1) / 255)]
            
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
        #sim = - np.log(mae/255.)
        self.fitness = sim
        return sim

# ---------------------------
# Clase del Algoritmo Genético
# ---------------------------
class GeneticAlgorithm:
    def __init__(self, target_path, canvas_size, n_triangles, pop_size, kids_size,
                 parents_selection_method, crossover_criteria, mutation_method, new_gen_creation_criteria, new_gen_selection_method, end_criteria, out_dir):
        self.target_path = target_path
        self.canvas_size = canvas_size
        self.n_triangles = n_triangles
        self.pop_size = pop_size
        self.kids_size = kids_size
        self.parents_selection_method = parents_selection_method
        self.crossover_method = crossover_criteria["method"]
        self.crossover_rate = crossover_criteria["crossover_rate"]
        self.mutation_method = mutation_method
        self.new_gen_creation_criteria = new_gen_creation_criteria
        self.new_gen_selection_method = new_gen_selection_method
        self.end_criteria = end_criteria
        self.out_dir = out_dir
        self.best_fit_history = []
        self.target_img = Image.open(target_path).convert('RGB').resize(canvas_size, Image.LANCZOS)
        self.target_rgb = np.array(self.target_img, dtype=np.uint8)

        self.gen = 0
        self.population = [Individual(n_triangles, canvas_size) for _ in range(pop_size)]
        self.best = None
        self.best_fitness = -1

    def select(self, individuals, k, method):
        """
        Selects k individuals based on the specified selection method.
        Supports 'elitism', 'roulette', 'boltzmann', 'ranking', 'universal', 'deterministic_tournament', and 'probabilistic_tournament' methods.
        - 'elitism' selects the top k individuals.
        - 'roulette' selects k individuals based on fitness proportionate selection.
        - 'ranking' selects k individuals based on their rank.
        - 'universal' selects k individuals using stochastic universal sampling.
        - 'deterministic_tournament'
        - 'probabilistic_tournament' 
        """

        # Elitismo
        if method["name"] == "elitism":
            # si k > N, devuelve solo los N individuos
            sorted_pop = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)
            return sorted_pop[:k]

        # Ruleta
        elif method["name"] == "roulette":
            total_fitness = sum(individual.fitness for individual in individuals)
            probs = [individual.fitness / total_fitness for individual in individuals]
            chosen = np.random.choice(individuals, size=k, replace=True, p=probs)
            return list(chosen)
    
        # Ranking
        elif method["name"] == "ranking":
            n = len(individuals)
            # Orden descendente por fitness → mejor = rank 0
            sorted_pop = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)

            # Pseudo-aptitud
            pseudo_fitness = [(n - rank) / n for rank in range(n)]

            total = sum(pseudo_fitness)
            probs = [pf / total for pf in pseudo_fitness]
            cumulative = np.cumsum(probs)

            chosen = []
            for _ in range(k):
                r = random.random()
                for i, cum in enumerate(cumulative):
                    if r <= cum:
                        chosen.append(sorted_pop[i])
                        break
            return chosen
        
        # Universal
        elif method["name"] == "universal":
            total_fitness = sum(individual.fitness for individual in individuals)
            probs = [individual.fitness / total_fitness for individual in individuals]
            cumulative = np.cumsum(probs)

            # Espaciado de 1/k entre cada puntero
            r = random.random() / k
            pointers = [(r + j / k) for j in range(k)]

            chosen = []
            for ptr in pointers:
                for i, cum in enumerate(cumulative):
                    if ptr <= cum:
                        chosen.append(individuals[i])
                        break
            return chosen
        
        # Boltzmann
        elif method["name"] == "boltzmann":
            T = method["Tc"] - (method["T0"] - method["Tc"]) * np.exp(-self.gen * method["decay"])  # temperatura decreciente
            exp_fitness = np.exp([individual.fitness / T for individual in individuals])
            probs = exp_fitness / np.sum(exp_fitness)
            chosen = np.random.choice(individuals, size=k, replace=True, p=probs)
            return list(chosen)
        
        # Torneo determinístico
        elif method["name"] == "deterministic_tournament":
            winners = []
            for _ in range(k):
                competitors = random.sample(individuals, method["M"])
                winner = max(competitors, key=lambda ind: ind.fitness)
                winners.append(winner)
            return winners
        
        # Torneo probabilístico
        elif method["name"] == "probabilistic_tournament":
            p = method["p"]  # prob. de elegir el mejor
            winners = []
            for _ in range(k):
                competitors = random.sample(individuals, 2)
                competitors.sort(key=lambda ind: ind.fitness, reverse=True)
                if random.random() < p:
                    winners.append(competitors[0])  # mejor
                else:
                    winners.append(competitors[-1])  # peor
            return winners

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
        
        # Uniform crossover: exchange genes based on given p
        elif self.crossover_method["name"] == "uniform":
            mask = np.random.rand(self.n_triangles) < self.crossover_method["p"]
            
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
                triangle = individual.genes[idx]
                for j in range(10):  # cada triángulo tiene 10 componentes
                    if random.random() < mutation_rate:
                        if j == 3:  # Alpha
                            triangle[j] = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)
                        else:  # RGB, X Y coord
                            triangle[j] = np.random.randint(0, 256)
                
                if MIN_AREA:
                    # asegurar que el triángulo resultante tenga área mínima
                    while triangle_area(triangle[4:]) < MIN_AREA:
                        xs = np.random.randint(0, 256, size=3)
                        ys = np.random.randint(0, 256, size=3)
                        triangle[4:] = np.array([*xs, *ys], dtype=np.uint8)
                                    
                individual.genes[idx] = triangle

        # Uniform multi-gene mutation: mutate each gene with a probability defined by mutation rate
        elif self.mutation_method["name"] == "uniform_multigene":
            for triangle in individual.genes:
                if random.random() < mutation_rate:
                    # randomly select one of the 10 components to mutate
                    j = np.random.randint(0, 10)
                    if j == 3:  # Alpha
                        triangle[j] = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)
                    else:  # RGB, X Y coord
                        triangle[j] = np.random.randint(0, 256)
                    
                    if MIN_AREA:
                        # asegurar que el triángulo resultante tenga área mínima
                        while triangle_area(triangle[4:]) < MIN_AREA:
                            xs = np.random.randint(0, 256, size=3)
                            ys = np.random.randint(0, 256, size=3)
                            triangle[4:] = np.array([*xs, *ys], dtype=np.uint8)

        elif self.mutation_method["name"] == "complete":
            flat = individual.genes.reshape(-1)
            mask = np.random.rand(flat.size) < mutation_rate
            flat[mask] = np.random.randint(0, 256, size=mask.sum(), dtype=np.uint8)
            # separar el random del alpha para que tome los valores correctos
            individual.genes = flat.reshape(individual.genes.shape)
    
            for i in range(flat.size):
                if mask[i]:
                    j = i % 10  # cada triángulo tiene 10 valores
                    if j == 3:  # Alpha
                        flat[i] = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)
                    else:  # RGB, X Y coord
                        flat[i] = np.random.randint(0, 256)

            individual.genes = flat.reshape(individual.genes.shape)
            
            # --- chequeo área mínima en todos los triángulos ---
            for triangle in individual.genes:
                if triangle_area(triangle[4:10]) < MIN_AREA:
                    while True:
                        coords = np.random.randint(0, 256, size=6)
                        if triangle_area(coords) >= MIN_AREA:
                            triangle[4:10] = coords
                            break
                        
    def create_new_generation(self, criteria, selection_method, current_population, kids):
        """
        Creates a new generation based on the specified criteria and selection method.
        Supports 'young_bias' and 'traditional' criteria.
        - 'young_bias': keeps all kids and selects the rest from the current population based on selection method.
        - 'traditional': selects the new generation from the combined pool of current population and kids based on selection method.
        """
        if criteria == "young_bias":
            if self.kids_size == self.pop_size:
                new_generation = kids
            elif self.kids_size > self.pop_size:
                new_generation = self.select(individuals=kids, k=self.pop_size, method=selection_method)
            else: # k < N
                new_generation = kids + self.select(individuals=current_population, k=self.pop_size-self.kids_size, method=selection_method)

        elif criteria == "traditional":
            new_generation = self.select(individuals=current_population+kids, k=self.pop_size, method=selection_method)

        return new_generation

    # Graficar evolución del fitness
    def plot_fitness(self, output_path):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.best_fit_history)+1), self.best_fit_history, marker="o")
        plt.title("Evolución del fitness")
        plt.xlabel("Generación")
        plt.ylabel("Best fitness")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

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

        while not self.evaluate_end_criteria(self.gen, start_time):
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

                # print best triangle areas
                areas = [triangle_area(triangle[4:]) for triangle in self.best.genes]
                print(f"New best fitness: {self.best_fitness:.4f} | Areas: min {min(areas):.1f}, max {max(areas):.1f}, avg {np.mean(areas):.1f}")

                # Mostrar las dos imágenes
                img1_rgb = np.array(gen_best.render().convert("RGB"))
                img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
                img2_bgr = cv2.cvtColor(self.target_rgb, cv2.COLOR_RGB2BGR)
                combined = np.hstack((img1_bgr, img2_bgr))
                cv2.imshow("Best vs Target", combined)
                if cv2.waitKey(100) & 0xFF == 27:  # ESC para salir
                    break
                
                # Guardar la mejor imagen cada vez que se mejora
                self.best.render().save(os.path.join(self.out_dir, f'best_gen{self.gen:05d}.png'))

            if self.gen % 10 == 0 or self.gen == 0: # cada 10 generaciones se imprime el fitness
                elapsed = time.time() - start_time
                print(f"Gen {self.gen:4d} | best {gen_best.fitness:.2f} | overall {self.best_fitness:.2f} | {elapsed:.1f}s")

            # Se seleccionan los k padres a cruzar
            parents_to_breed = self.select(individuals=self.population, k=self.kids_size, method=self.parents_selection_method)

            # Nueva población
            kids = []

            # Recorrer de a pares de padres (k/2 pares)
            for i in range(0, len(parents_to_breed), 2):
                parent_a = parents_to_breed[i]
                parent_b = parents_to_breed[i+1]

                # Cruzar 2 padres -> genera 2 hijos
                child1, child2 = self.crossover(parent_a, parent_b)

                # Mutación
                self.mutate(child1)
                self.mutate(child2)

                child1.evaluate_fitness(self.target_rgb)
                child2.evaluate_fitness(self.target_rgb)

                kids.append(child1)
                kids.append(child2)

            # Selección de la nueva generación:
            self.population = self.create_new_generation(criteria=self.new_gen_creation_criteria, selection_method=self.new_gen_selection_method, current_population=self.population, kids=kids)

            self.gen += 1

        cv2.destroyAllWindows()

        print("Finalizado. Mejor similarity:", np.round(self.best_fitness, 3))
        # Grafica evolución del fitness
        self.plot_fitness(os.path.join(self.out_dir, "fitness_evolution.png"))
        # Guarda la mejor compresion final
        self.best.render().save(os.path.join(self.out_dir, "best_final.png"))
        # Guardar un txt con los triangulos del mejor individuo
        with open(os.path.join(self.out_dir, "best_triangles.txt"), "w") as f:
            for triangle in self.best.genes:
                f.write(','.join(map(str, triangle)) + '\n')