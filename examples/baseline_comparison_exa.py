from cilpy.problem.constrained import G01
from cilpy.runner import ExperimentRunner
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.de import DE
from cilpy.solver.ga import GA
from cilpy.solver.pso import PSO

DIMENSION = 500
POP_SIZE = 60

problems_to_run = [
    G01(),
]

solver_configs = [
    # CCLS with PSO
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_PSO",
            "objective_solver_class": PSO,
            "objective_solver_params": {
                "name": "objective_pso",
                "swarm_size": POP_SIZE,
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
    # CCLS with DE
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_DE",
            "objective_solver_class": DE,
            "objective_solver_params": {
                "name": "objective_de",
                "population_size": POP_SIZE,
                "crossover_rate": 0.9,
                "f_weight": 0.8,
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
    # CCLS with GA
    {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_GA",
            "objective_solver_class": GA,
            "objective_solver_params": {
                "name": "objective_ga",
                "population_size": POP_SIZE,
                "crossover_rate": 0.8,
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
    # Plain PSO
    {
        "class": PSO,
        "params": {
            "name": "PSO",
            "swarm_size": POP_SIZE,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
        },
    },
    # Plain DE
    {
        "class": DE,
        "params": {
            "name": "DE",
            "population_size": POP_SIZE,
            "crossover_rate": 0.9,
            "f_weight": 0.8,
        },
    },
    # Plain GA
    {
        "class": GA,
        "params": {
            "name": "GA",
            "population_size": POP_SIZE,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "tournament_size": 3,
        },
    },
]

runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=5,
    max_iterations=1000,
)
runner.run_experiments()
