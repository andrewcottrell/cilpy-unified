from cilpy.problem.unconstrained import Ackley, Quadratic, Rastrigin, Sphere
from cilpy.runner import ExperimentRunner
from cilpy.solver.de import DE
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.ga import GA
from cilpy.solver.pso import PSO

DIMENSION = 100
NUM_GROUPS = 6
SUB_POPULATION_SIZE = 10
FULL_POPULATION_SIZE = NUM_GROUPS * SUB_POPULATION_SIZE

# --- 1. Define the problems ---
problems_to_run = [
    Sphere(dimension=DIMENSION, domain=(-100, 100)),
    Quadratic(dimension=DIMENSION, domain=(-100, 100)),
    Ackley(dimension=DIMENSION, domain=(-32, 32)),
    Rastrigin(dimension=DIMENSION, domain=(-5.12, 5.12)),
]

# --- 2. Define solver configurations ---
solver_configs = [
    # --- PSO ---
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": {
                "name": "sub_pso",
                "swarm_size": SUB_POPULATION_SIZE,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
        },
    },
    {
        "class": PSO,
        "params": {
            "name": "PSO_baseline",
            "swarm_size": FULL_POPULATION_SIZE,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
        },
    },
    # --- GA ---
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CGA",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": GA,
            "sub_solver_params": {
                "name": "sub_ga",
                "population_size": SUB_POPULATION_SIZE,
                "crossover_rate": 0.9,
                "mutation_rate": 0.1,
                "tournament_size": 3,
            },
        },
    },
    {
        "class": GA,
        "params": {
            "name": "GA_baseline",
            "population_size": FULL_POPULATION_SIZE,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
            "tournament_size": 3,
        },
    },
    # --- DE ---
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CDE",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": DE,
            "sub_solver_params": {
                "name": "sub_de",
                "population_size": SUB_POPULATION_SIZE,
                "crossover_rate": 0.9,
                "f_weight": 0.8,
            },
        },
    },
    {
        "class": DE,
        "params": {
            "name": "DE_baseline",
            "population_size": FULL_POPULATION_SIZE,
            "crossover_rate": 0.9,
            "f_weight": 0.8,
        },
    },
]

# --- 3. Run ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=3,
    max_iterations=1000,
)
runner.run_experiments()
