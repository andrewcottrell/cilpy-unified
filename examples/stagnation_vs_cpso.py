from cilpy.problem.unconstrained import Quadratic, Sphere, Ackley, Rastrigin
from cilpy.runner import ExperimentRunner
from cilpy.solver.de import DE
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.pso import PSO
from cilpy.solver.ga import GA
from cilpy.solver.decomposition.drg import DynamicRandomGroupingSolver
from cilpy.solver.decomposition.trigger import StagnationTrigger

DIMENSION = 100
NUM_GROUPS = 6
SUB_SWARM_SIZE = 10
TOTAL_PARTICLES = NUM_GROUPS * SUB_SWARM_SIZE

PATIENCE = 20

TOLERANCE = 1e-2

# --- 1. Define the problem ---
problems_to_run = [
    Sphere(dimension=DIMENSION, domain=(-5.12, 5.12)),
    Quadratic(dimension=DIMENSION, domain=(-100, 100)),
    Ackley(dimension=DIMENSION, domain=(-32, 32)),
    Rastrigin(dimension=DIMENSION, domain=(-5.12, 5.12)),
]


solver_configs = [

    # Stagnation triggered PSO
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "STAG_PSO",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": {
                "name": "sub_pso",
                "swarm_size": SUB_SWARM_SIZE,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
            "regroup_trigger": StagnationTrigger(
                patience=PATIENCE, tolerance=TOLERANCE),
        },
    },

    # CPSO
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO_ref",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": {
                "name": "sub_pso",
                "swarm_size": SUB_SWARM_SIZE,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
        },
    },

    # Stagnation triggered GA
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "STAG_GA",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": GA,
            "sub_solver_params": {
                "name": "sub_ga",
                "population_size": SUB_SWARM_SIZE,
                "crossover_rate": 0.9,
                "mutation_rate": 0.1,
                "tournament_size": 3,
            },
            "regroup_trigger": StagnationTrigger(
                patience=PATIENCE, tolerance=TOLERANCE),
        },
    },

    # CGA
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CGA_ref",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": GA,
            "sub_solver_params": {
                "name": "sub_ga",
                "population_size": SUB_SWARM_SIZE,
                "crossover_rate": 0.9,
                "mutation_rate": 0.1,
                "tournament_size": 3,
            },
        },
    },

    # Stagnation triggered DE
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "STAG_DE",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": DE,
            "sub_solver_params": {
                "name": "sub_de",
                "population_size": SUB_SWARM_SIZE,
                "crossover_rate": 0.9,
                "f_weight": 0.8,
            },
            "regroup_trigger": StagnationTrigger(
                patience=PATIENCE, tolerance=TOLERANCE),
        },
    },
    # CDE
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CDE_ref",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": DE,
            "sub_solver_params": {
                "name": "sub_de",
                "population_size": SUB_SWARM_SIZE,
                "crossover_rate": 0.9,
                "f_weight": 0.8,
            },
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
