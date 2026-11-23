# examples/ccmgpso_exa.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.dynamic_multi_objective import DTNK4
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO

# --- 1. Define the Problems ---
problems_to_run = [
    DTNK4(tau_t=25)
]

# --- 2. Define the Solvers and their parameters ---
# Note: The 'problem' parameter is omitted here as the runner will assign it.
solver_configs = [
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCMGPSO",
            "objective_solver_class": MGPSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": 50,
                "refresh_mode": "clean",
                "sentry_mode": "archive",
                "w": 0.72,
                "c1": 1.49,
                "c2": 1.49,
                "c3": 1.49,
            },
            "multiplier_solver_params": {
                "swarm_size": 30,
                "w": 0.40,
                "c1": 1.20,
                "c2": 1.20,
            },
            "multiplier_max": 1000.0,
            "multiplier_immigrants": 0.3,
            "multiplier_steps": 20,
            "reuse_evaluations": True,
            "archive_strategy": "strict",
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 1
max_iter = 250

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
