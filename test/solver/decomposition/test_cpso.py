# test/solver/decomposition/test_cpso.py
"""
Unit tests for StandardCooperativeSolver.
"""

from cilpy.problem.unconstrained import Sphere
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.de import DE
from cilpy.solver.pso import PSO

class TestInitialisation:
    """Tests for creating the solver."""

    def test_solver_can_be_created_without_errors(self):
        """The solver can be created with valid settings."""

        solver = make_solver()
        assert solver is not None

    def test_correct_number_of_sub_solvers_are_created(self):
        """One sub-solver is created per group."""

        solver = make_solver(num_groups=3)
        assert len(solver._sub_solvers) == 3

    def test_correct_number_of_sub_problems_are_created(self):
        """One sub-problem is created per group."""

        solver = make_solver(num_groups=3)
        assert len(solver._sub_problems) == 3

    def test_num_groups_exceeding_dimension_raises_value_error(self):
        """More groups than dimensions raises a ValueError."""

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
    """Tests for running iterations."""

    def test_step_runs_without_raising_an_exception(self):
        """One step runs without errors."""

        solver = make_solver()
        solver.step()

    def test_multiple_steps_run_without_raising_an_exception(self):
        """Ten steps in a row run without errors."""

        solver = make_solver()

        for _ in range(10):
            solver.step()

    def test_fitness_history_grows_by_one_entry_per_step(self):
        """Every step adds one entry to the best fitness history."""

        solver = make_solver()

        solver.step()
        solver.step()
        solver.step()

        assert len(solver._best_fitness_history) == 3

class TestContextUpdate:
    """Tests for when a group's best is written into the context."""

    def test_only_accept_improvements_is_off_by_default(self):
        """CPSO does not use only_accept_improvements."""

        solver = make_solver()

        assert solver._only_accept_improvements is False

    def test_worse_update_is_still_accepted(self):
        """Without only_accept_improvements, a worse candidate is still written into the context."""

        solver = make_solver()

        assert solver._accepts_context_update(candidate_fitness=5.0, current_context_fitness=3.0) is True

class TestGetResult:
    """Tests for the result returned by the solver."""

    def test_get_result_returns_a_list_with_one_entry(self):
        """The result is a list with one solution."""

        solver = make_solver()

        result = solver.get_result()

        assert isinstance(result, list)
        assert len(result) == 1

    def test_returned_solution_has_the_correct_dimension(self):
        """The returned solution holds every dimension of the problem."""

        dimension = 12
        solver = make_solver(dimension=dimension)

        best_solution, best_evaluation = solver.get_result()[0]

        assert len(best_solution) == dimension

    def test_fitness_is_a_float(self):
        """The returned fitness is a float."""

        solver = make_solver()

        best_solution, best_evaluation = solver.get_result()[0]

        assert isinstance(best_evaluation.fitness, float)

class TestGetPopulation:
    """Tests for the full-dimension population."""

    def test_get_population_returns_the_correct_number_of_individuals(self):
        """The population holds every individual of every sub-solver."""

        num_groups = 3
        swarm_size = 5
        solver = make_solver(num_groups=num_groups, swarm_size=swarm_size)

        population = solver.get_population()

        assert len(population) == num_groups * swarm_size

    def test_each_individual_in_population_has_the_correct_dimension(self):
        """Every individual is expanded to the full problem dimension."""

        dimension = 12
        solver = make_solver(dimension=dimension, num_groups=3, swarm_size=4)

        population = solver.get_population()

        for individual in population:
            assert len(individual) == dimension

# --- Helper functions ---

def make_solver(dimension=12, num_groups=3, swarm_size=5):
    """Returns a CPSO with PSO sub-solvers on a small Sphere problem."""

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
