from cilpy.problem.unconstrained import Quadratic, Sphere, Ackley, Rastrigin, Rosenbrock
from cilpy.runner import ExperimentRunner
from cilpy.solver.de import DE
from cilpy.solver.ga import GA
from cilpy.solver.pso import PSO

DIMENSION = 100

NUM_GROUPS = 6
SUB_SWARM_SIZE = 10
TOTAL_PARTICLES = NUM_GROUPS * SUB_SWARM_SIZE

# --- 1. Define the problem ---
problems_to_run = [
    Rosenbrock(dimension=DIMENSION, domain=(-30, 30))
]

# --- 2. Define the solver configurations ---
solver_configs = [

    # Baseline PSO
    {
        "class": PSO,
        "params": {
            "name": "PSO_baseline",
            "swarm_size": TOTAL_PARTICLES,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
        },
    },

    # Baseline GA
    {
        "class": GA,
        "params": {
            "name": "GA_baseline",
            "population_size": TOTAL_PARTICLES,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
            "tournament_size": 3,
        },
    },

    # Baseline DE
    {
        "class": DE,
        "params": {
            "name": "DE_baseline",
            "population_size": TOTAL_PARTICLES,
            "crossover_rate": 0.9,
            "f_weight": 0.8,
        },
    },
]

# --- 3. Run ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=30,
    max_iterations=5000,
)
runner.run_experiments()
