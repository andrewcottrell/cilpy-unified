# examples/dcpso_examples.py
from cilpy.problem.unconstrained import Rosenbrock
from cilpy.runner import ExperimentRunner
from cilpy.solver.decomposition.grouping import (
    EvenSplitter, RandomGrouping, RandomGroupSizeGrouping, RandomSplitter, StaticEvenGrouping)
from cilpy.solver.decomposition.dcpso import DecompositionCooperativeSolver
from cilpy.solver.decomposition.trigger import (
    EvaluationIntervalTrigger, FixedIntervalTrigger, StagnationTrigger, restructure_interval)
from cilpy.solver.pso import PSO

DIMENSION = 100
MAX_EVALUATIONS = 1000 * DIMENSION
SPLIT_FACTOR = 2


EVALUATION_INTERVAL = restructure_interval(MAX_EVALUATIONS, DIMENSION, SPLIT_FACTOR)
START_GROUPS = 5

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

    # DCPSO defaults
    {
        "class": DecompositionCooperativeSolver,
        "params": {
            "name": "DCPSO_defaults",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "max_evaluations": MAX_EVALUATIONS,
        },
    },

    # --- Groupings ---

    # DCPSO with static grouping
    {
        "class": DecompositionCooperativeSolver,
        "params": {
            "name": "DCPSO_static_grouping",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "num_groups": START_GROUPS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_splitter": EvenSplitter(split_factor=SPLIT_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # DCPSO with random grouping
    {
        "class": DecompositionCooperativeSolver,
        "params": {
            "name": "DCPSO_random_grouping",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "num_groups": START_GROUPS,
            "grouping_strategy": RandomGrouping(),
            "group_splitter": EvenSplitter(split_factor=SPLIT_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # DCPSO with random group size grouping
    {
        "class": DecompositionCooperativeSolver,
        "params": {
            "name": "DCPSO_random_group_size",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": RandomGroupSizeGrouping(group_sizes=[20, 50, 100]),
            "group_splitter": EvenSplitter(split_factor=SPLIT_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # --- Splitters ---

    # DCPSO with random splitter
    {
        "class": DecompositionCooperativeSolver,
        "params": {
            "name": "DCPSO_random_splitter",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_splitter": RandomSplitter(split_factor=SPLIT_FACTOR),
            "regroup_trigger": EvaluationIntervalTrigger(interval=EVALUATION_INTERVAL),
        },
    },

    # --- Triggers ---

    # DCPSO with fixed interval trigger
    {
        "class": DecompositionCooperativeSolver,
        "params": {
            "name": "DCPSO_fixed_interval",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_splitter": EvenSplitter(split_factor=SPLIT_FACTOR),
            "regroup_trigger": FixedIntervalTrigger(interval=20),
        },
    },

    # DCPSO with stagnation trigger
    {
        "class": DecompositionCooperativeSolver,
        "params": {
            "name": "DCPSO_stagnation",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "grouping_strategy": StaticEvenGrouping(),
            "group_splitter": EvenSplitter(split_factor=SPLIT_FACTOR),
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
