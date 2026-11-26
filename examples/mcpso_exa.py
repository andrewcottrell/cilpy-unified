# examples/mcpso_exa.py
"""Compares MCPSO (even and random merger) with CPSO and plain PSO on
100-D Rastrigin.
"""

from cilpy.problem.unconstrained import Rastrigin
from cilpy.runner import ExperimentRunner
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.decomposition.mcpso import MergingCooperativeSolver
from cilpy.solver.decomposition.grouping import RandomMerger
from cilpy.solver.pso import PSO

DIMENSION = 100
NUM_GROUPS = 6
SUB_SWARM_SIZE = 10
TOTAL_PARTICLES = NUM_GROUPS * SUB_SWARM_SIZE
MAX_ITERATIONS = 1000

SUB_PSO_PARAMS = {
    "name": "sub_pso",
    "swarm_size": SUB_SWARM_SIZE,
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

    # MCPSO
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "max_iterations": MAX_ITERATIONS,
        },
    },

    # MCPSO with random grouping
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO_random",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "group_merger": RandomMerger(merge_factor=2),
            "max_iterations": MAX_ITERATIONS,
        },
    },

    # CPSO with a fixed number of groups
    {
        "class": StandardCooperativeSolver,
        "params": {
            "name": "CPSO",
            "num_groups": NUM_GROUPS,
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
        },
    },

    # Normal PSO
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
]

# --- 3. Run ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=3,
    max_iterations=MAX_ITERATIONS,
)
runner.run_experiments()
