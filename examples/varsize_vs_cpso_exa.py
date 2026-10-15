from cilpy.problem.unconstrained import Quadratic, Sphere, Ackley, Rastrigin
from cilpy.runner import ExperimentRunner
from cilpy.solver.de import DE
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.pso import PSO
from cilpy.solver.ga import GA
from cilpy.solver.decomposition.drg import DynamicRandomGroupingSolver
from cilpy.solver.decomposition.grouping import RandomGroupSizeGrouping
from cilpy.solver.decomposition.trigger import FixedIntervalTrigger

DIMENSION = 500
SUB_SWARM_SIZE = 10

GROUP_SIZES = [5, 10, 50, 100, 250]
NUM_GROUPS = 6
INITIAL_NUM_GROUPS = 6
REGROUP_INTERVAL = 50

# --- 1. Define the problem ---
problems_to_run = [
    Sphere(dimension=DIMENSION, domain=(-5.12, 5.12)),
    Quadratic(dimension=DIMENSION, domain=(-100, 100)),
    Ackley(dimension=DIMENSION, domain=(-32, 32)),
    Rastrigin(dimension=DIMENSION, domain=(-5.12, 5.12)),
]

# --- 2. Define the solver configurations ---
solver_configs = [

    # Variable group size PSO
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "VARSIZE_PSO",
            "num_groups": INITIAL_NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": {
                "name": "sub_pso",
                "swarm_size": SUB_SWARM_SIZE,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
            "grouping_strategy": RandomGroupSizeGrouping(group_sizes=GROUP_SIZES),
            "regroup_trigger": FixedIntervalTrigger(interval=REGROUP_INTERVAL),
        },
    },

    # CPSO
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO_fixed",
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

    # Variable group size GA
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "VARSIZE_GA",
            "num_groups": INITIAL_NUM_GROUPS,
            "sub_solver_class": GA,
            "sub_solver_params": {
                "name": "sub_ga",
                "population_size": SUB_SWARM_SIZE,
                "crossover_rate": 0.9,
                "mutation_rate": 0.1,
                "tournament_size": 3,
            },
            "grouping_strategy": RandomGroupSizeGrouping(group_sizes=GROUP_SIZES),
            "regroup_trigger": FixedIntervalTrigger(interval=REGROUP_INTERVAL),
        },
    },

    # CGA
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CGA_fixed",
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

    # Variable group size DE
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "VARSIZE_DE",
            "num_groups": INITIAL_NUM_GROUPS,
            "sub_solver_class": DE,
            "sub_solver_params": {
                "name": "sub_de",
                "population_size": SUB_SWARM_SIZE,
                "crossover_rate": 0.9,
                "f_weight": 0.8,
            },
            "grouping_strategy": RandomGroupSizeGrouping(group_sizes=GROUP_SIZES),
            "regroup_trigger": FixedIntervalTrigger(interval=REGROUP_INTERVAL),
        },
    },
    # CDE
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CDE_fixed",
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
    num_runs=10,
    max_iterations=5000,
)
runner.run_experiments()
