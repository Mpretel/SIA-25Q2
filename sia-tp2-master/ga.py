#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random, os, time
import cv2

from constants import MIN_ALPHA, MAX_ALPHA, MIN_AREA, MAX_AREA, SEED, RGB, DELTA

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    
def triangle_area(coords):
    """
    Computes the area of a triangle given its 3 vertices
    coords = [x1, y1, x2, y2, x3, y3]
    """
    x1, y1, x2, y2, x3, y3 = map(float, coords) 
    return abs(0.5 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))

# ---------------------------
# Class that represents an individual solution
# ---------------------------
class Individual:
    # Initializes the class with n_triangles and canvas size.
    # Each triangle has 10 genes: RGBA (4) + 3 vertices (6 coords: x,y)
    def __init__(self, n_triangles, canvas_size, genes=None):
        self.n_triangles = n_triangles
        self.canvas_size = canvas_size
        # Genes are initialized randomly if not provided...
        if genes is None:
            # Create an empty array for all triangles
            self.genes = np.zeros((n_triangles, 10), dtype=np.uint8)

            for i in range(n_triangles):
                if RGB:
                    # Initialize color as RGB + alpha
                    r, g, b = np.random.randint(0, 256, size=3)
                else:
                    # Alternative: HSV color space
                    h, s, v = np.random.randint(0, 256, size=3)
                a = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)

                # Generate 3 vertices (x,y) ∈ [0, 255] respecting min/max area
                if MIN_AREA or MAX_AREA:
                    coords = np.random.randint(0, 256, size=6)
                    while triangle_area(coords) < MIN_AREA or triangle_area(coords) > MAX_AREA:
                        coords = np.random.randint(0, 256, size=6)
                else:
                    coords = np.random.randint(0, 256, size=6)

                # Store triangle genes
                if RGB:
                    self.genes[i] = np.array([r, g, b, a, *coords], dtype=np.uint8)
                else:
                    self.genes[i] = np.array([h, s, v, a, *coords], dtype=np.uint8)

        # ...otherwise copy from a previous generation
        else:
            self.genes = genes.astype(np.uint8)
            
        # Fitness starts as None
        self.fitness = None

    # Renders the individual's image (drawing all its triangles)
    def render(self):
        W, H = self.canvas_size
        # White canvas
        base = Image.new('RGBA', (W, H), (255, 255, 255, 255))

        # Draw triangles in descending order of alpha (opaque on top)
        sorted_genes = sorted(self.genes, key=lambda triangle: triangle[3], reverse=True)

        for triangle in sorted_genes:
            triangle = triangle.astype(np.int32)
            if RGB:
                # First 4 values are RGBA
                r, g, b, a = [int(x) for x in triangle[:4]]
            else:
                # Convert HSV to RGB (OpenCV expects H ∈ [0,179])
                h, s, v, a = [int(x) for x in triangle[:4]]
                h = int(h * 179 / 255)
                r, g, b = [int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0][0]]

            # Next 6 values are 3 vertices (x,y)
            xs = [int(triangle[4] * (W-1) / 255),
                int(triangle[6] * (W-1) / 255),
                int(triangle[8] * (W-1) / 255)]

            ys = [int(triangle[5] * (H-1) / 255),
                int(triangle[7] * (H-1) / 255),
                int(triangle[9] * (H-1) / 255)]
            
            poly = list(zip(xs, ys))
            # Draw the triangle on a transparent layer and merge
            layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer, 'RGBA')
            draw.polygon(poly, fill=(r, g, b, a))
            base = Image.alpha_composite(base, layer)
        return base

    def evaluate_fitness(self, target_rgb):
        """
        Computes the fitness of the individual as the similarity to the target image.
        Fitness is defined as 1 - normalized Mean Absolute Error (MAE) between 
        the rendered image and the target image, so higher values indicate better similarity.
        
        Parameters:
        - target_rgb: NumPy array of shape (H, W, 3) representing the target image in RGB
        
        Returns:
        - sim: fitness value ∈ [0, 1], where 1 means perfect match
        """
        # Render the individual's image and convert to RGB NumPy array
        rendered = np.array(self.render().convert('RGB'), dtype=np.uint8)
        # Convert arrays to int32 to avoid overflow when computing differences
        r = rendered.astype(np.int32)
        t = target_rgb.astype(np.int32)
        # Compute Mean Absolute Error between rendered and target image
        mae = np.mean(np.abs(r - t))
        # Convert MAE to similarity in [0,1]
        sim = 1.0 - (mae / 255.0)
        # Store fitness in the individual
        self.fitness = sim
        return sim

# ---------------------------
# Class for the Genetic Algorithm
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
       # Initialize population with random individuals
        self.population = [Individual(n_triangles, canvas_size) for _ in range(pop_size)]
        self.best = None
        self.best_fitness = -1

    def select(self, individuals, k, method):
        """
        Selection operator.
        Chooses `k` individuals from the population based on the specified selection method.
        Methods supported:
        - 'elitism'               : select the top k individuals.
        - 'roulette'              : select based on fitness-proportionate probability.
        - 'ranking'               : select based on rank-derived probability.
        - 'universal'             : stochastic universal sampling (evenly spaced pointers).
        - 'boltzmann'             : probability changes over time based on a temperature function.
        - 'deterministic_tournament': pick the best among M sampled individuals.
        - 'probabilistic_tournament': pick best of 2 with probability p, otherwise the worst.
        """
        # Elitism
        if method["name"] == "elitism":
            # Sort population by fitness and return the top k
            sorted_pop = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)
            return sorted_pop[:k]
    
        # Roulette wheel selection
        elif method["name"] == "roulette":
            # Probability proportional to individual fitness
            total_fitness = sum(individual.fitness for individual in individuals)
            probs = [individual.fitness / total_fitness for individual in individuals]
            chosen = np.random.choice(individuals, size=k, replace=True, p=probs)
            return list(chosen)
        
        # Ranking selection
        elif method["name"] == "ranking":
            # Assign probability based on sorted rank instead of raw fitness
            n = len(individuals)
            sorted_pop = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)
            pseudo_fitness = [(n - rank) / n for rank in range(n)]  # rank-based score
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
        
        # Stochastic Universal Sampling
        elif method["name"] == "universal":
            total_fitness = sum(individual.fitness for individual in individuals)
            probs = [individual.fitness / total_fitness for individual in individuals]
            cumulative = np.cumsum(probs)
    
            # Evenly spaced selection points
            r = random.random() / k
            pointers = [(r + j / k) for j in range(k)]
    
            chosen = []
            for ptr in pointers:
                for i, cum in enumerate(cumulative):
                    if ptr <= cum:
                        chosen.append(individuals[i])
                        break
            return chosen
        
        # Boltzmann selection
        elif method["name"] == "boltzmann":
            # Temperature decreases over time
            T = method["Tc"] - (method["T0"] - method["Tc"]) * np.exp(-self.gen * method["decay"])
            exp_fitness = np.exp([individual.fitness / T for individual in individuals])
            probs = exp_fitness / np.sum(exp_fitness)
            chosen = np.random.choice(individuals, size=k, replace=True, p=probs)
            return list(chosen)
        
        # Deterministic tournament
        elif method["name"] == "deterministic_tournament":
            winners = []
            for _ in range(k):
                competitors = random.sample(individuals, method["M"])  # pick M competitors
                winner = max(competitors, key=lambda ind: ind.fitness) # best wins
                winners.append(winner)
            return winners
        
        # Probabilistic tournament
        elif method["name"] == "probabilistic_tournament":
            p = method["p"]  # probability of picking the best
            winners = []
            for _ in range(k):
                competitors = random.sample(individuals, 2) # pick 2 competitors
                competitors.sort(key=lambda ind: ind.fitness, reverse=True)
                if random.random() < p:
                    winners.append(competitors[0])  # choose best
                else:
                    winners.append(competitors[-1]) # choose worst
            return winners

    def crossover(self, parent_a, parent_b):
        """
        Performs crossover between two parent individuals to produce two offspring.
        
        Supports:
        - 'one-point': exchange genes after a single random crossover point.
        - 'uniform'  : exchange genes based on a probability mask.
        
        If crossover does not occur (based on crossover rate), children are exact copies of parents.
        """
        # Decide if crossover happens
        if random.random() > self.crossover_rate:
            return (
                Individual(self.n_triangles, self.canvas_size, genes=parent_a.genes.copy()),
                Individual(self.n_triangles, self.canvas_size, genes=parent_b.genes.copy())
            )
        
        # One-point crossover
        if self.crossover_method["name"] == "one-point":
            point = np.random.randint(1, self.n_triangles)
            child1_genes = np.concatenate([parent_a.genes[:point], parent_b.genes[point:]])
            child2_genes = np.concatenate([parent_b.genes[:point], parent_a.genes[point:]])
            return (
                Individual(self.n_triangles, self.canvas_size, genes=child1_genes),
                Individual(self.n_triangles, self.canvas_size, genes=child2_genes)
            )
        
        # Uniform crossover
        elif self.crossover_method["name"] == "uniform":
            mask = np.random.rand(self.n_triangles) < self.crossover_method["p"]  # gene-wise mask
    
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
        Mutates an individual based on the selected mutation method.
        
        Supported methods:
        - 'limited_multigene' : mutate M genes selected randomly with mutation_rate.
        - 'uniform_multigene' : mutate each gene independently with mutation_rate.
        - 'complete'           : full random mutation over all genes based on mutation_rate.
        
        Mutation affects:
        - Color channels (RGB or HSV)
        - Alpha channel
        - Vertex coordinates (XY)
        
        Ensures that triangle areas respect MIN_AREA and MAX_AREA if defined.
        """
        mutation_rate = self.mutation_method["mutation_rate"]
    
        # Limited multi-gene mutation
        if self.mutation_method["name"] == "limited_multigene":
            M = min(self.mutation_method["M"], self.n_triangles - 1)
            gene_indices = np.random.choice(self.n_triangles, M, replace=False)
    
            for idx in gene_indices:
                triangle = individual.genes[idx]
                for j in range(10):  # each triangle has 10 values
                    if random.random() < mutation_rate:
                        if j == 3:  # Alpha channel
                            triangle[j] = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)
                        elif j > 3:  # X/Y coordinates
                            triangle[j] = np.random.randint(0, 256)
                        else:  # R/G/B channels
                            if DELTA:
                                triangle[j] = np.clip(int(triangle[j]) + np.random.randint(-DELTA, DELTA), 0, 255)
                            else:
                                triangle[j] = np.random.randint(0, 256)
    
                # Ensure triangle area is within limits
                if MIN_AREA or MAX_AREA:
                    while triangle_area(triangle[4:]) < MIN_AREA or triangle_area(triangle[4:]) > MAX_AREA:
                        xs = np.random.randint(0, 256, size=3)
                        ys = np.random.randint(0, 256, size=3)
                        triangle[4:] = np.array([*xs, *ys], dtype=np.uint8)
                individual.genes[idx] = triangle
    
        # Uniform multi-gene mutation
        elif self.mutation_method["name"] == "uniform_multigene":
            for triangle in individual.genes:
                if random.random() < mutation_rate:
                    j = np.random.randint(0, 10)  # select one component
                    if j == 3:  # Alpha
                        triangle[j] = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)
                    elif j > 3:  # XY
                        triangle[j] = np.random.randint(0, 256)
                    else:  # RGB/HSV
                        if DELTA:
                            triangle[j] = np.clip(int(triangle[j]) + np.random.randint(-DELTA, DELTA), 0, 255)
                        else:
                            triangle[j] = np.random.randint(0, 256)
    
                    # Check area constraints
                    if MIN_AREA or MAX_AREA:
                        while triangle_area(triangle[4:]) < MIN_AREA or triangle_area(triangle[4:]) > MAX_AREA:
                            xs = np.random.randint(0, 256, size=3)
                            ys = np.random.randint(0, 256, size=3)
                            triangle[4:] = np.array([*xs, *ys], dtype=np.uint8)
    
        # Complete mutation
        elif self.mutation_method["name"] == "complete":
            flat = individual.genes.reshape(-1)
            mask = np.random.rand(flat.size) < mutation_rate
            flat[mask] = np.random.randint(0, 256, size=mask.sum(), dtype=np.uint8)
    
            # Correct alpha values
            for i in range(flat.size):
                if mask[i]:
                    j = i % 10
                    if j == 3:
                        flat[i] = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1)
                    else:
                        flat[i] = np.random.randint(0, 256)
    
            individual.genes = flat.reshape(individual.genes.shape)
    
            # Ensure area constraints for all triangles
            for triangle in individual.genes:
                if triangle_area(triangle[4:10]) < MIN_AREA or triangle_area(triangle[4:10]) > MAX_AREA:
                    while True:
                        coords = np.random.randint(0, 256, size=6)
                        if MIN_AREA <= triangle_area(coords) <= MAX_AREA:
                            triangle[4:10] = coords
                            break
                        
    def create_new_generation(self, criteria, selection_method, current_population, kids):
        """
        Generates a new population for the next generation.
        
        Parameters:
        - criteria: 'young_bias' keeps all kids, then fills remaining from current population
                    'traditional' selects new population from combined pool of current + kids
        - selection_method: method used to select individuals when required
        - current_population: current generation
        - kids: offspring produced by crossover and mutation
        
        Returns:
        - new_generation: list of individuals for the next generation
        """
        if criteria == "young_bias":
            if self.kids_size == self.pop_size:
                new_generation = kids
            elif self.kids_size > self.pop_size:
                # more kids than population size → select top N kids
                new_generation = self.select(individuals=kids, k=self.pop_size, method=selection_method)
            else:
                # fill remaining slots from current population
                new_generation = kids + self.select(individuals=current_population, k=self.pop_size - self.kids_size, method=selection_method)
        elif criteria == "traditional":
            # select from combined pool
            new_generation = self.select(individuals=current_population + kids, k=self.pop_size, method=selection_method)
    
        return new_generation
    
    def plot_fitness(self, output_path):
        """
        Plots the evolution of the best fitness over generations.
        
        Parameters:
        - output_path: path to save the plot image (PNG)
        
        The function plots the fitness progression, adds grid and labels, and saves the figure.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.best_fit_history)+1), self.best_fit_history, marker="o")
        plt.title("Fitness Evolution")
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
    
    def evaluate_end_criteria(self, gen, start_time):
        """
        Checks whether the stopping condition for the genetic algorithm has been met.
        
        Parameters:
        - gen: current generation number
        - start_time: timestamp when the GA started
        
        Supported end criteria:
        - 'generations': stop after a fixed number of generations
        - 'fitness'    : stop when best fitness reaches target value
        - 'time'       : stop when elapsed time exceeds allowed limit
        
        Returns:
        - True if the algorithm should stop, False otherwise
        """
        if self.end_criteria["name"] == "generations":
            return gen >= self.end_criteria["value"]
        elif self.end_criteria["name"] == "fitness":
            return self.best_fitness >= self.end_criteria["value"]
        elif self.end_criteria["name"] == "time":
            elapsed = time.time() - start_time
            return elapsed >= self.end_criteria["value"]
        else:
            raise ValueError("Unknown end criteria")
    
    def run(self):
        """
        Main execution loop of the genetic algorithm for image approximation.
        
        Steps performed per generation:
        1. Evaluate fitness of each individual if not already evaluated.
        2. Track the best individual of the generation and update global best.
           - Display current best vs. target image using OpenCV.
           - Save the best individual image if improvement occurs.
        3. Print progress every 10 generations.
        4. Select parents for breeding using the configured selection method.
        5. Perform crossover and mutation to generate offspring.
        6. Evaluate fitness of all offspring.
        7. Create the next generation according to the new generation criteria.
        8. Increment generation counter.
        9. Repeat until stopping criteria are met.
        
        After completion:
        - Close OpenCV windows.
        - Save final best image and fitness evolution plot.
        - Save best triangles configuration and fitness per generation.
        """
        start_time = time.time()        
    
        while not self.evaluate_end_criteria(self.gen, start_time):
            # --- Evaluate population fitness ---
            for individual in self.population:
                if individual.fitness is None:
                    individual.evaluate_fitness(self.target_rgb)
    
            # --- Track the best individual of this generation ---
            gen_best = max(self.population, key=lambda individual: individual.fitness)
            self.best_fit_history.append(gen_best.fitness)
    
            if gen_best.fitness > self.best_fitness:
                self.best = gen_best
                self.best_fitness = gen_best.fitness
    
                # Display best vs target image side by side
                img1_rgb = np.array(gen_best.render().convert("RGB"))
                img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
                img2_bgr = cv2.cvtColor(self.target_rgb, cv2.COLOR_RGB2BGR)
                combined = np.hstack((img1_bgr, img2_bgr))
                cv2.imshow("Best vs Target", combined)
                if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit
                    break
                
                # Save the improved best image
                self.best.render().save(os.path.join(self.out_dir, f'best_gen{self.gen:05d}.png'))
    
            # Print progress every 10 generations
            if self.gen % 10 == 0 or self.gen == 0:
                elapsed = time.time() - start_time
                print(f"Gen {self.gen:4d} | best {gen_best.fitness:.2f} | overall {self.best_fitness:.2f} | {elapsed:.1f}s")
    
            # --- Parent selection for breeding ---
            parents_to_breed = self.select(individuals=self.population, k=self.kids_size, method=self.parents_selection_method)
    
            # --- Generate offspring ---
            kids = []
            for i in range(0, len(parents_to_breed), 2):
                parent_a = parents_to_breed[i]
                parent_b = parents_to_breed[i+1]
    
                # Perform crossover to generate two children
                child1, child2 = self.crossover(parent_a, parent_b)
    
                # Apply mutation
                self.mutate(child1)
                self.mutate(child2)
    
                # Evaluate offspring fitness
                child1.evaluate_fitness(self.target_rgb)
                child2.evaluate_fitness(self.target_rgb)
    
                kids.append(child1)
                kids.append(child2)
    
            # --- Create new generation ---
            self.population = self.create_new_generation(
                criteria=self.new_gen_creation_criteria,
                selection_method=self.new_gen_selection_method,
                current_population=self.population,
                kids=kids
            )
    
            self.gen += 1
    
        # --- Finalization ---
        cv2.destroyAllWindows()
    
        print("Finished. Best similarity:", np.round(self.best_fitness, 3))
    
        # Save fitness evolution plot
        self.plot_fitness(os.path.join(self.out_dir, "fitness_evolution.png"))
    
        # Save the final best image
        self.best.render().save(os.path.join(self.out_dir, "best_final.png"))
    
        # Save triangle coordinates of the best individual
        with open(os.path.join(self.out_dir, "best_triangles.txt"), "w") as f:
            for triangle in self.best.genes:
                f.write(','.join(map(str, triangle)) + '\n')
    
        # Save fitness history as CSV
        with open(os.path.join(self.out_dir, "fitness_evolution.csv"), "w") as f:
            f.write("generation,best_fitness\n")
            for gen, fit in enumerate(self.best_fit_history):
                f.write(f"{gen},{fit}\n")
