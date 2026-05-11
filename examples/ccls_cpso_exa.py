from cilpy.problem.constrained import ConstrainedSphere
from cilpy.runner import ExperimentRunner
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.pso import PSO

DIMENSION = 100
NUM_GROUPS = 6
SUB_SWARM_SIZE = 10

# --- 1. Define the problem ---
problems_to_run = [
    ConstrainedSphere(dimension=DIMENSION, domain=(-100, 100)),
]

# --- 2. Define solver configurations ---
solver_configs = [
    # CCLS with StandardCooperativeSolver as the objective solver
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_CPSO",
            "objective_solver_class": StandardCooperativeSolver,
            "objective_solver_params": {
                "name": "CPSO_objective",
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
            "multiplier_solver_class": PSO,
            "multiplier_solver_params": {
                "name": "multiplier_pso",
                "swarm_size": 20,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
        },
    },
    # CCLS with plain PSO
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_PSO_baseline",
            "objective_solver_class": PSO,
            "objective_solver_params": {
                "name": "objective_pso",
                "swarm_size": NUM_GROUPS * SUB_SWARM_SIZE,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
            "multiplier_solver_class": PSO,
            "multiplier_solver_params": {
                "name": "multiplier_pso",
                "swarm_size": 20,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
        },
    },
]

# --- 3. Run ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=5,
    max_iterations=10000,
)
runner.run_experiments()
