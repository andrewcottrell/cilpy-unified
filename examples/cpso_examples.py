# examples/cpso_examples.py
from cilpy.problem.unconstrained import Rastrigin
from cilpy.runner import ExperimentRunner
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.decomposition.grouping import (
    RandomGrouping, RandomGroupSizeGrouping, StaticEvenGrouping)
from cilpy.solver.pso import PSO

DIMENSION = 100
MAX_EVALUATIONS = 1000 * DIMENSION
NUM_GROUPS = 10

SUB_PSO_PARAMS = {
    "name": "sub_pso",
    "swarm_size": 10,
    "w": 0.7298,
    "c1": 1.49618,
    "c2": 1.49618,
}

# --- 1. Define the problem ---
problems_to_run = [
    Rastrigin(dimension=DIMENSION, domain=(-5.12, 5.12)),
]

# --- 2. Define the solver configurations ---
solver_configs = [

    # CPSO defaults
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO_defaults",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
        },
    },

    # --- Groupings ---

    # CPSO with static grouping
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO_static_grouping",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
        },
    },

    # CPSO with random grouping
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO_random_grouping",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGrouping(),
        },
    },

    # CPSO with random group size grouping
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO_random_group_size",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGroupSizeGrouping(group_sizes=[5, 10, 25]),
        },
    },

]

# --- 3. Run ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=3,
    max_evaluations=MAX_EVALUATIONS,
)
runner.run_experiments()
