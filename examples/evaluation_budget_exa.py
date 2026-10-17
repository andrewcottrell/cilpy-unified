from cilpy.problem.unconstrained import Rastrigin
from cilpy.runner import ExperimentRunner
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.decomposition.dcpso import DecompositionCooperativeSolver
from cilpy.solver.decomposition.mcpso import MergingCooperativeSolver
from cilpy.solver.pso import PSO

DIMENSION = 100
NUM_GROUPS = 6
SUB_SWARM_SIZE = 10
TOTAL_PARTICLES = NUM_GROUPS * SUB_SWARM_SIZE

MAX_EVALUATIONS = 3000 * DIMENSION

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

    # DCPSO spaces its splits over the evaluation budget
    {
        "class": DecompositionCooperativeSolver,
        "params": {
            "name": "DCPSO",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "max_evaluations": MAX_EVALUATIONS,
        },
    },

    # MCPSO spaces its merges over the evaluation budget
    {
        "class": MergingCooperativeSolver,
        "params": {
            "name": "MCPSO",
            "sub_solver_class": PSO,
            "sub_solver_params": SUB_PSO_PARAMS,
            "max_evaluations": MAX_EVALUATIONS,
        },
    },

    # CPSO
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
    max_evaluations=MAX_EVALUATIONS,
)
runner.run_experiments()
