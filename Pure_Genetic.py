# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Setting random seed for reproducibility
np.random.seed(42)

# Define a class for individual members in the genetic algorithm
class Individual:
    def __init__(self, genes):
        """
        Initialize an Individual object.

        Parameters:
        - genes (array): Array representing the genes of the individual.
        """
        self.genes = genes

    def calculate_fitness(self):
        """
        Calculate the fitness of the individual based on a specific function.

        Returns:
        - float: Calculated fitness value.
        """
        x = np.linspace(0, 1, 100)
        E1 = np.abs(1/2 - self.genes[0])
        dydx = 2 * self.genes[2] * x + self.genes[1]
        y = self.genes[2] * x**2 + self.genes[1] * x + self.genes[0]
        estimated_y = y + dydx + E1
        exact_y = 1/4 * x**2 - 1/2 * x + 1/2
        fit_exact_y = np.mean(np.abs(exact_y - estimated_y))
        return 1 / fit_exact_y

# Define a function for crossover between two parents
def crossover(parent1, parent2):
    """
    Perform crossover between two parent individuals.

    Parameters:
    - parent1 (Individual): First parent individual.
    - parent2 (Individual): Second parent individual.

    Returns:
    - Individual: Child individual resulting from crossover.
    """
    crossover_point = np.random.randint(len(parent1.genes))
    child_genes = np.concatenate((parent1.genes[:crossover_point], parent2.genes[crossover_point:]))
    return Individual(child_genes)

# Define a function for mutation of an individual
def mutate(individual, mutation_rate):
    """
    Mutate the genes of an individual based on a mutation rate.

    Parameters:
    - individual (Individual): Individual to be mutated.
    - mutation_rate (float): Rate of mutation.

    Returns:
    - Individual: Mutated individual.
    """
    mutation_mask = (np.random.rand(len(individual.genes)) < mutation_rate).astype(int)
    mutation_values = np.random.randn(len(individual.genes))
    individual.genes += mutation_mask * mutation_values
    return individual


# Define a function for roulette wheel selection of individuals
def roulette_wheel_selection(population):
    """
    Perform roulette wheel selection of individuals from a given population.

    Parameters:
    - population (list): List of Individual objects.

    Returns:
    - list: List of selected individuals based on roulette wheel selection.
    """
    total_fitness = np.sum([ind.calculate_fitness() for ind in population])
    probabilities = [ind.calculate_fitness() / total_fitness for ind in population]
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    selected_individuals = [population[i] for i in selected_indices]
    return selected_individuals


# Define the main genetic algorithm function
def genetic_algorithm(population_size, number_of_iterations, mutation_rate, crossover_probability):
    """
    Implement a genetic algorithm to optimize individuals based on a specific function.

    Parameters:
    - population_size (int): Size of the population.
    - number_of_iterations (int): Number of iterations for the algorithm.
    - mutation_rate (float): Rate of mutation.
    - crossover_probability (float): Probability of crossover.

    Returns:
    - None: Results are printed and plotted.
    """
    population = [Individual(np.random.randn(3)) for _ in range(population_size)]

    loss_progress = []

    for iteration in range(1, number_of_iterations + 1):  # Start from 1 for better conditional check
        selected_parents = roulette_wheel_selection(population)

        new_population = []

        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]

            if np.random.rand() < crossover_probability:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent1, parent2)
            else:
                child1 = parent1
                child2 = parent2

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population

        best_solution = max(population, key=lambda x: x.calculate_fitness())
        loss_progress.append(1 / best_solution.calculate_fitness() * 100)

        # Check if the current iteration is a multiple of 1000
        if iteration % 1000 == 0 and iteration != number_of_iterations :
            print(f"Results after {iteration} iterations:")
            print(f"Estimated Parameters: w[0]={best_solution.genes[0]}, w[1]={best_solution.genes[1]}, w[2]={best_solution.genes[2]}")
            print(f"Fitness: {best_solution.calculate_fitness()}")
            print(f"Mean Absolute Error: {1 / best_solution.calculate_fitness()}")
            print(f"Mean Squared Error: {(1 / best_solution.calculate_fitness())**2}")

    # Print and plot final results
    print("\nFinal Results:")
    print(f"Estimated Parameters: w[0]={best_solution.genes[0]}, w[1]={best_solution.genes[1]}, w[2]={best_solution.genes[2]}")
    print(f"Fitness: {best_solution.calculate_fitness()}")
    print(f"Mean Absolute Error: {1 / best_solution.calculate_fitness()}")
    print(f"Mean Squared Error: {(1 / best_solution.calculate_fitness())**2}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(loss_progress)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss through the generations")

    plt.subplot(1, 3, 2)
    x = np.linspace(0, 1, 100)
    E1 = np.abs(1/2 - best_solution.genes[0])
    dydx = 2 * best_solution.genes[2] * x + best_solution.genes[1]
    y = best_solution.genes[2] * x**2 + best_solution.genes[1] * x + best_solution.genes[0]
    estimated_y = y + dydx + E1
    exact_y = 1/4 * x**2 - 1/2 * x + 1/2
    plt.plot(x, estimated_y, color='red', label='Estimated')
    plt.plot(x, exact_y, color='blue', label='Exact')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Estimated vs Exact function")

    plt.subplot(1, 3, 3)
    plt.plot(x, np.abs(estimated_y - exact_y))
    plt.xlabel("x")
    plt.ylabel("Absolute error")
    plt.title("Absolute error")
    plt.tight_layout()
    plt.show()

# Set parameters for the genetic algorithm
population_size = 50
number_of_iterations = 2000
mutation_rate = 0.1
crossover_probability = 0.7

# Run the genetic algorithm
genetic_algorithm(population_size, number_of_iterations, mutation_rate, crossover_probability)