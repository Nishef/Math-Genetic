# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Setting random seed for reproducibility
np.random.seed(42)

# Define a class for Streamer objects
class Streamer:
    def __init__(self, genes, electrons, radius):
        """
        Initialize a Streamer object.

        Parameters:
        - genes (array): Array representing the genes of the streamer.
        - electrons (int): Number of electrons associated with the streamer.
        - radius (float): Radius parameter for the streamer.
        """
        self.genes = genes.copy()
        self.electrons = electrons
        self.radius = radius
        self.last_position = self.genes.copy()

# Define a function to calculate fitness
def calculate_fitness(genes, x):
    """
    Calculate the fitness of genes based on a given function.

    Parameters:
    - genes (array): Array representing the genes of the streamer.
    - x (float): Input value for the function.

    Returns:
    - float: Calculated fitness value.
    """
    E1 = np.abs(1/2 - genes[0])
    dydx = 2 * genes[2] * x + genes[1]
    y = genes[2] * x**2 + genes[1] * x + genes[0]
    estimated_y = y + dydx + E1
    exact_y = 1/4 * x**2 - 1/2 * x + 1/2
    loss = np.abs(estimated_y - exact_y)
    return 1 / (1 + loss)

# Define a function to update the streamer based on specific conditions
def update_streamer(streamer, x, best_fitness, beta, radius_multiplier, M):
    """
    Update the streamer's genes based on specific conditions.

    Parameters:
    - streamer (Streamer): Streamer object to be updated.
    - x (array): Input values for the function.
    - best_fitness (float): Current best fitness value.
    - beta (float): Beta parameter for the update.
    - radius_multiplier (float): Radius multiplier for the update.
    - M (int): Number of random points to consider.

    Returns:
    - float: Updated best fitness value.
    """
    F_nt = calculate_fitness(streamer.genes, x[-1])
    F_nt_1 = calculate_fitness(streamer.genes, x[-2])
    F_n0 = calculate_fitness(streamer.genes, x[0])

    best_fitness = max(best_fitness, F_nt)

    distance = np.linalg.norm(streamer.genes - streamer.last_position)
    streamer.last_position = streamer.genes.copy()

    lambda_val = beta * x[-1] * (1 - np.exp(-(F_nt - F_nt_1))) + (1 - beta) * x[-1] * np.exp(-(F_nt - best_fitness))

    if distance < streamer.radius * radius_multiplier:
        random_points = np.random.normal(loc=0, scale=streamer.radius, size=(M, len(streamer.genes)))
        random_points += streamer.genes

        random_fitness = calculate_fitness(random_points, x[-1])
        best_random_fitness = np.max(random_fitness)
        best_random_point = random_points[np.argmax(random_fitness)]

        if best_random_fitness > F_nt:
            streamer.genes = best_random_point
            streamer.electrons = streamer.electrons * np.exp(((F_nt - F_nt_1) / F_n0) * x[-1]) + lambda_val

    else:
        streamer.genes = streamer.genes * np.exp(((F_nt - F_nt_1) / F_n0) * x[-1]) + lambda_val
        best_random_fitness = 0  # Assign a default value if the condition is not met


    return best_fitness

# Define a function to check for forking events
def forking_event(streamer, CV_ratio):
    """
    Check if a forking event should occur for the streamer.

    Parameters:
    - streamer (Streamer): Streamer object.
    - CV_ratio (float): Coefficient of variation ratio.

    Returns:
    - bool: True if a forking event should occur, False otherwise.
    """
    return streamer.electrons > 2 * CV_ratio and np.random.rand() < 1 - 2 * CV_ratio / streamer.electrons

# Define a function to eliminate a streamer based on specific conditions
def eliminate_streamer(streamer, r_max, lb, ub):
    """
    Check if a streamer should be eliminated based on specific conditions.

    Parameters:
    - streamer (Streamer): Streamer object.
    - r_max (float): Maximum radius for elimination.
    - lb (float): Lower bound for genes.
    - ub (float): Upper bound for genes.

    Returns:
    - bool: True if the streamer should be eliminated, False otherwise.
    """
    return np.linalg.norm(streamer.genes) > r_max or not np.all((lb <= streamer.genes) & (streamer.genes <= ub))


# Set parameters
N = 200
E0_n_range = [500, 1500]
CV_ratio = 0.1
r_range = [0.1, 10]
M = N**2
best_streamer = None
lb = -5.0
ub = 5.0
dimensions = 3
max_iterations = 10
beta = 0.25
radius_multiplier = 0.5
r_max = 1.0

# Additional initialization
loss_progress = []  # Rename fitness_progress to loss_progress

# Initialize population of streamers
population = [Streamer(np.random.uniform(lb, ub, dimensions), np.random.randint(*E0_n_range), np.random.uniform(*r_range)) for _ in range(N)]

# Main ERSA Loop
best_fitness = 0
for iteration in range(max_iterations):
    x_values = np.linspace(lb, ub, 10)
    fitness_values = []

    for i, streamer in enumerate(population):
        fitness_row = [calculate_fitness(streamer.genes, x_value) for x_value in x_values]
        fitness_values.append(fitness_row)

        current_fitness = max(fitness_row)
        best_fitness = max(best_fitness, current_fitness)

        if current_fitness == best_fitness:
            best_streamer = streamer  # Update best_streamer

        if forking_event(streamer, CV_ratio):
            new_streamer = Streamer(streamer.genes.copy(), streamer.electrons // 2, streamer.radius)
            population.append(new_streamer)
            streamer.electrons //= 2
            new_streamer.electrons = streamer.electrons

        best_fitness = update_streamer(streamer, x_values, best_fitness, beta, radius_multiplier, M)

    # Eliminate streamers based on specific conditions
    population = [streamer for streamer in population if not eliminate_streamer(streamer, r_max, lb, ub)]

    loss_progress.append(best_fitness)  # Update the name to loss_progress

    r_max_list = [streamer.radius * np.exp((streamer.electrons - CV_ratio) * (1.0 - 1.0 / CV_ratio)) for streamer in population]
    max_index = min(len(population), N)  # Ensure that the index is within bounds
    r_max_list = r_max_list[:max_index]
    population = population[:max_index]

    # Termination condition check
    if iteration == max_iterations - 1:
        break

    population.sort(key=lambda s: calculate_fitness(s.genes, x_values[0]), reverse=True)

    print(f"Iteration {iteration + 1}: Fitness = {calculate_fitness(best_streamer.genes, x_values[0])}")
    print("Genes:", best_streamer.genes)

# Define a function to find the best streamer in the population
def find_best(population):
    """
    Find the best streamer in the given population.

    Parameters:
    - population (list): List of Streamer objects.

    Returns:
    - Streamer: Best streamer based on fitness.
    """
    if not population:
        return print("No Answer, Please Run The Code Again.")  # Handle the case where the population is empty
    return max(population, key=lambda s: calculate_fitness(s.genes, x_values[0]))

# Find the best solution in the population
best_solution = find_best(population)

# Calculate estimated and exact y values
estimated_y = best_solution.genes[2] * x_values**2 + best_solution.genes[1] * x_values + best_solution.genes[0]
exact_y = 1/4 * x_values**2 - 1/2 * x_values + 1/2

# Calculate fitness metrics
Fitness = calculate_fitness(best_solution.genes, x_values[0]) * 100
MAE = 1 / Fitness
MSE = MAE ** 2

# Print the results
print(f"Estimated Parameters: w[0]={best_solution.genes[0]}, w[1]={best_solution.genes[1]}, w[2]={best_solution.genes[2]}")
print(f'Exact Parameters: w[2]=0.25, w[1]=-0.5, w[0]=0.5')
print(f"Fitness: {Fitness}")
print(f"Mean Absolute Error: {MAE}")
print(f"Mean Squared Error: {MSE}")

# Plotting
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss_progress)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss through the generations")

plt.subplot(1, 3, 2)
plt.plot(x_values, estimated_y, color='red', label='Estimated')
plt.plot(x_values, exact_y, color='blue', label='Exact')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Estimated vs Exact function")

plt.subplot(1, 3, 3)
plt.plot(x_values, np.abs(estimated_y - exact_y))
plt.xlabel("x")
plt.ylabel("Absolute error")
plt.title("Absolute error")
plt.tight_layout()
plt.show()
