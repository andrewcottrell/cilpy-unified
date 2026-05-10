from cilpy.problem.unconstrained import Sphere
from cilpy.runner import ExperimentRunner
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.pso import PSO

NUM_GROUPS = 10
SUB_SWARM_SIZE = 5
TOTAL_PARTICLES = NUM_GROUPS * SUB_SWARM_SIZE

# --- 1. Define the problem ---
problems_to_run = [
    Sphere(dimension=100, domain=(-100, 100)),
]

# --- 2. Define the solver configurations ---
solver_configs = [
    # Decomposition solver with 10 groups × 5 particles each
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
    # Normal PSO with the same total number of particles
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
    num_runs=5,
    max_iterations=2000,
)
runner.run_experiments()
