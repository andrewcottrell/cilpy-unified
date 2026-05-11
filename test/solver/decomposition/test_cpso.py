# test/solver/decomposition/test_cpso.py
"""
Unit tests for StandardCooperativeSolver.
"""

from cilpy.problem.unconstrained import Sphere
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.pso import PSO

class TestInitialisation:

    def test_solver_can_be_created_without_errors(self):
        
        solver = make_solver()
        assert solver is not None

    def test_correct_number_of_sub_solvers_are_created(self):

        solver = make_solver(num_groups=3)
        assert len(solver._sub_solvers) == 3

    def test_correct_number_of_sub_problems_are_created(self):

        solver = make_solver(num_groups=3)
        assert len(solver._sub_problems) == 3

    def test_num_groups_exceeding_dimension_raises_value_error(self):

        problem = Sphere(dimension=4, domain=(-5.0, 5.0))
        sub_solver_params = {
            "name": "sub_pso",
            "swarm_size": 5,
            "w": 0.7,
            "c1": 1.5,
            "c2": 1.5,
        }

        try:
            StandardCooperativeSolver(
                problem=problem,
                name="bad_solver",
                num_groups=10,
                sub_solver_class=PSO,
                sub_solver_params=sub_solver_params,
            )
            raised = False
        except ValueError:
            raised = True

        assert raised, "Expected a ValueError when num_groups > dimension"

class TestStep:

    def test_step_runs_without_raising_an_exception(self):

        solver = make_solver()
        solver.step()

    def test_multiple_steps_run_without_raising_an_exception(self):

        solver = make_solver()

        for _ in range(10):
            solver.step()

    def test_fitness_history_grows_by_one_entry_per_step(self):

        solver = make_solver()

        solver.step()
        solver.step()
        solver.step()

        assert len(solver._best_fitness_history) == 3

class TestGetResult:

    def test_get_result_returns_a_list_with_one_entry(self):

        solver = make_solver()

        result = solver.get_result()

        assert isinstance(result, list)
        assert len(result) == 1

    def test_returned_solution_has_the_correct_dimension(self):

        dimension = 12
        solver = make_solver(dimension=dimension)

        best_solution, best_evaluation = solver.get_result()[0]

        assert len(best_solution) == dimension

    def test_fitness_is_a_float(self):

        solver = make_solver()

        best_solution, best_evaluation = solver.get_result()[0]

        assert isinstance(best_evaluation.fitness, float)

class TestGetPopulation:

    def test_get_population_returns_the_correct_number_of_individuals(self):

        num_groups = 3
        swarm_size = 5
        solver = make_solver(num_groups=num_groups, swarm_size=swarm_size)

        population = solver.get_population()

        assert len(population) == num_groups * swarm_size

    def test_each_individual_in_population_has_the_correct_dimension(self):

        dimension = 12
        solver = make_solver(dimension=dimension, num_groups=3, swarm_size=4)

        population = solver.get_population()

        for individual in population:
            assert len(individual) == dimension

# --- Helper functions ---

def make_solver(dimension=12, num_groups=3, swarm_size=5):

    problem = Sphere(dimension=dimension, domain=(-5.0, 5.0))

    sub_solver_params = {
        "name": "sub_pso",
        "swarm_size": swarm_size,
        "w": 0.7298,
        "c1": 1.49618,
        "c2": 1.49618,
    }

    solver = StandardCooperativeSolver(
        problem=problem,
        name="test_cpso",
        num_groups=num_groups,
        sub_solver_class=PSO,
        sub_solver_params=sub_solver_params,
    )

    return solver