# examples/mcpso_combinations_exa.py
from cilpy.problem.unconstrained import Rastrigin
from cilpy.runner import ExperimentRunner
from cilpy.solver.decomposition.grouping import (
    EvenMerger, RandomGrouping, RandomGroupSizeGrouping, RandomMerger, StaticEvenGrouping)
from cilpy.solver.decomposition.mcpso import MergingCooperativeSolver
from cilpy.solver.decomposition.trigger import (
    EvaluationIntervalTrigger, FixedIntervalTrigger, NeverTrigger, StagnationTrigger,
    restructure_interval)
from cilpy.solver.pso import PSO

DIMENSION = 100
MAX_EVALUATIONS = 1000 * DIMENSION
MERGE_FACTOR = 2


EVALUATION_INTERVAL = restructure_interval(MAX_EVALUATIONS, DIMENSION, MERGE_FACTOR)
START_GROUPS = 20

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

    # MCPSO, all strategies explicit (static grouping, even merger, evaluation interval)
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_merger": EvenMerger(merge_factor=MERGE_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # MCPSO, defaults
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO_defaults",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "max_evaluations": MAX_EVALUATIONS,
        },
    },

    # --- Groupings ---

    # MCPSO with static grouping
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO_static_grouping",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "num_groups": START_GROUPS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_merger": EvenMerger(merge_factor=MERGE_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # MCPSO with random grouping
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO_random_grouping",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "num_groups": START_GROUPS,
            "grouping_strategy": RandomGrouping(),
            "group_merger": EvenMerger(merge_factor=MERGE_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # MCPSO with random group size grouping
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO_random_group_size",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGroupSizeGrouping(group_sizes=[1, 2, 5]),
            "group_merger": EvenMerger(merge_factor=MERGE_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # --- Mergers ---

    # MCPSO with random merger
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO_random_merger",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_merger": RandomMerger(merge_factor=MERGE_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # --- Triggers ---

    # MCPSO with fixed interval trigger
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO_fixed_interval",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_merger": EvenMerger(merge_factor=MERGE_FACTOR),
            "regroup_trigger": FixedIntervalTrigger(interval=20),
        },
    },

    # MCPSO with stagnation trigger
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO_stagnation",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_merger": EvenMerger(merge_factor=MERGE_FACTOR),
            "regroup_trigger": StagnationTrigger(patience=10, tolerance=1e-4),
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
