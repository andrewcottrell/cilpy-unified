# examples/ccls_cpso_exa.py
"""Runs CCLS with cooperative solvers (CPSO, CGA and CDE) as the objective
solver on 100-D ConstrainedSphere, next to CCLS with plain PSO, GA and DE.

Shows that the cooperative solvers also work inside the constrained framework.
"""

from cilpy.problem.constrained import ConstrainedSphere
from cilpy.runner import ExperimentRunner
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.de import DE
from cilpy.solver.ga import GA
from cilpy.solver.pso import PSO

DIMENSION = 100
NUM_GROUPS = 6
SUB_POPULATION_SIZE = 10
FULL_POPULATION_SIZE = NUM_GROUPS * SUB_POPULATION_SIZE

# --- 1. Define the problem ---
problems_to_run = [
    ConstrainedSphere(dimension=DIMENSION, domain=(-100, 100)),
]

# --- 2. Define solver configurations ---
solver_configs = [
    # --- PSO ---
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
                    "swarm_size": SUB_POPULATION_SIZE,
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
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_PSO_baseline",
            "objective_solver_class": PSO,
            "objective_solver_params": {
                "name": "objective_pso",
                "swarm_size": FULL_POPULATION_SIZE,
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
    # --- GA ---
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_CGA",
            "objective_solver_class": StandardCooperativeSolver,
            "objective_solver_params": {
                "name": "CGA_objective",
                "num_groups": NUM_GROUPS,
                "sub_solver_class": GA,
                "sub_solver_params": {
                    "name": "sub_ga",
                    "population_size": SUB_POPULATION_SIZE,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1,
                    "tournament_size": 3,
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
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_GA_baseline",
            "objective_solver_class": GA,
            "objective_solver_params": {
                "name": "objective_ga",
                "population_size": FULL_POPULATION_SIZE,
                "crossover_rate": 0.9,
                "mutation_rate": 0.1,
                "tournament_size": 3,
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
    # --- DE ---
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_CDE",
            "objective_solver_class": StandardCooperativeSolver,
            "objective_solver_params": {
                "name": "CDE_objective",
                "num_groups": NUM_GROUPS,
                "sub_solver_class": DE,
                "sub_solver_params": {
                    "name": "sub_de",
                    "population_size": SUB_POPULATION_SIZE,
                    "crossover_rate": 0.9,
                    "f_weight": 0.8,
                    "tournament_size": 3,
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
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_DE_baseline",
            "objective_solver_class": DE,
            "objective_solver_params": {
                "name": "objective_de",
                "population_size": FULL_POPULATION_SIZE,
                "crossover_rate": 0.9,
                "f_weight": 0.8,
                "tournament_size": 3,
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
