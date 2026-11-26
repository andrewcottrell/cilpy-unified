# examples/drg_examples.py
"""Shows every grouping and trigger the dynamic random grouping solver (DRG)
can use, on 100-D Rosenbrock with a budget of 1000 x D evaluations.
"""

from cilpy.problem.unconstrained import Rosenbrock
from cilpy.runner import ExperimentRunner
from cilpy.solver.decomposition.drg import DynamicRandomGroupingSolver
from cilpy.solver.decomposition.grouping import (
    RandomGrouping, RandomGroupSizeGrouping, StaticEvenGrouping)
from cilpy.solver.decomposition.trigger import (
    EvaluationIntervalTrigger, FixedIntervalTrigger, StagnationTrigger)
from cilpy.solver.pso import PSO

DIMENSION = 100
MAX_EVALUATIONS = 1000 * DIMENSION
NUM_GROUPS = 10


EVALUATION_INTERVAL = MAX_EVALUATIONS // 20

SUB_PSO_PARAMS = {
    "name": "sub_pso",
    "swarm_size": 10,
    "w": 0.7298,
    "c1": 1.49618,
    "c2": 1.49618,
}

# --- 1. Define the problem ---
problems_to_run = [
    Rosenbrock(dimension=DIMENSION, domain=(-30, 30)),
]

# --- 2. Define the solver configurations ---
solver_configs = [

    # DRG defaults
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_defaults",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
        },
    },

    # --- Groupings ---

    # DRG with static grouping (same groups after every regroup, so the same as CPSO)
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_static_grouping",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
            "regroup_trigger": FixedIntervalTrigger(interval=50),
        },
    },

    # DRG with random grouping
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_random_grouping",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGrouping(),
            "regroup_trigger": FixedIntervalTrigger(interval=50),
        },
    },

    # DRG with random group size grouping
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_random_group_size",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGroupSizeGrouping(group_sizes=[5, 10, 25, 50]),
            "regroup_trigger": FixedIntervalTrigger(interval=50),
        },
    },

    # --- Triggers ---

    # DRG with fixed interval trigger
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_fixed_interval",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGrouping(),
            "regroup_trigger": FixedIntervalTrigger(interval=20),
        },
    },

    # DRG with evaluation interval trigger
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_evaluation_interval",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGrouping(),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # DRG with stagnation trigger
    {
        "class": DynamicRandomGroupingSolver,
        "params": {
            "name": "DRG_stagnation",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGrouping(),
            "regroup_trigger": StagnationTrigger(patience=10, tolerance=1e-2),
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
