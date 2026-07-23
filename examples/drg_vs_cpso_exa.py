from cilpy.problem.unconstrained import Quadratic, Sphere, Ackley, Rastrigin
from cilpy.runner import ExperimentRunner
from cilpy.solver.de import DE
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.pso import PSO
from cilpy.solver.ga import GA
from cilpy.solver.decomposition.drg import DynamicRandomGroupingSolver
from cilpy.solver.decomposition.trigger import FixedIntervalTrigger

DIMENSION = 500
NUM_GROUPS = 6
SUB_SWARM_SIZE = 10
TOTAL_PARTICLES = NUM_GROUPS * SUB_SWARM_SIZE

# --- 1. Define the problem ---
problems_to_run = [
    Sphere(dimension=DIMENSION, domain=(-100, 100)),
    Quadratic(dimension=DIMENSION, domain=(-100, 100)),
    Ackley(dimension=DIMENSION, domain=(-32, 32)),
    Rastrigin(dimension=DIMENSION, domain=(-5.12, 5.12)),
]

# --- 2. Define the solver configurations ---
solver_configs = [
    
    # DRG PSO
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_PSO",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": {
                "name": "sub_pso",
                "swarm_size": SUB_SWARM_SIZE,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
            "regroup_trigger": FixedIntervalTrigger(interval=1),
        },
    },

    # CPSO
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO",
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

    # DRG GA
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_GA",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": GA,
            "sub_solver_params": {
                "name": "sub_ga",
                "population_size": SUB_SWARM_SIZE,
                "crossover_rate": 0.9,
                "mutation_rate": 0.1,
                "tournament_size": 3,
            },
            "regroup_trigger": FixedIntervalTrigger(interval=1),
        },
    },

    # CGA
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CGA",
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

    # DRG DE
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_DE",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": DE,
            "sub_solver_params": {
                "name": "sub_de",
                "population_size": SUB_SWARM_SIZE,
                "crossover_rate": 0.9,
                "f_weight": 0.8,
            },
            "regroup_trigger": FixedIntervalTrigger(interval=1),
        },
    },
    # CDE
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CDE",
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
    num_runs=3,
    max_iterations=10000,
)
runner.run_experiments()
