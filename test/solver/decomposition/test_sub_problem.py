# test/solver/decomposition/test_sub_problem.py
"""
Unit tests for SubSpaceProblem.
"""

from unittest.mock import MagicMock
from cilpy.problem import Evaluation, Problem
from cilpy.solver.decomposition.context import ContextVector
from cilpy.solver.decomposition.sub_problem import SubSpaceProblem

class TestInitialisation:

    def test_dimension_equals_the_number_of_indices_in_the_group(self):

        original_problem = make_temp_problem(dimension=6)
        context = make_context(6, [[0, 1, 2], [3, 4, 5]])

        sub_problem = SubSpaceProblem(
            original_problem, context, group_idx=0, dim_indices=[0, 1, 2]
        )

        assert sub_problem.dimension == 3

    def test_bounds_are_extracted_for_the_correct_indices(self):

        original_problem = make_temp_problem(dimension=6)
        original_problem.bounds = (
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            [1.0,  2.0,  3.0,  4.0,  5.0,  6.0],
        )
        context = make_context(6, [[1, 3], [0, 2, 4, 5]])

        sub_problem = SubSpaceProblem(
            original_problem, context, group_idx=0, dim_indices=[1, 3]
        )

        assert sub_problem.bounds[0] == [-2.0, -4.0]
        assert sub_problem.bounds[1] == [2.0, 4.0]


class TestEvaluate:

    def test_sub_solution_values_land_at_the_correct_global_indices(self):

        original_problem = make_temp_problem(dimension=6)
        context = make_context(6, [[0, 2, 4], [1, 3, 5]])

        sub_problem = SubSpaceProblem(
            original_problem, context, group_idx=0, dim_indices=[0, 2, 4]
        )
        sub_problem.evaluate([10.0, 20.0, 30.0])

        full_vector_passed_to_original = original_problem.evaluate.call_args[0][0]
        assert full_vector_passed_to_original[0] == 10.0
        assert full_vector_passed_to_original[2] == 20.0
        assert full_vector_passed_to_original[4] == 30.0

    def test_positions_not_owned_by_this_group_come_from_the_context(self):

        original_problem = make_temp_problem(dimension=6)
        context = make_context(6, [[0, 2, 4], [1, 3, 5]])
        context.update_group(group_idx=1, sub_solution=[7.0, 8.0, 9.0])

        sub_problem = SubSpaceProblem(
            original_problem, context, group_idx=0, dim_indices=[0, 2, 4]
        )
        sub_problem.evaluate([1.0, 2.0, 3.0])

        full_vector_passed_to_original = original_problem.evaluate.call_args[0][0]
        assert full_vector_passed_to_original[1] == 7.0
        assert full_vector_passed_to_original[3] == 8.0
        assert full_vector_passed_to_original[5] == 9.0

    def test_evaluation_result_is_returned_unchanged(self):

        original_problem = make_temp_problem(dimension=4)
        expected_evaluation = Evaluation(
            fitness=42.0,
            constraints_inequality=[0.5],
        )
        original_problem.evaluate.return_value = expected_evaluation

        context = make_context(4, [[0, 1], [2, 3]])
        sub_problem = SubSpaceProblem(
            original_problem, context, group_idx=0, dim_indices=[0, 1]
        )

        returned_evaluation = sub_problem.evaluate([0.0, 0.0])
        assert returned_evaluation is expected_evaluation

    def test_evaluate_does_not_write_values_back_into_the_context(self):

        original_problem = make_temp_problem(dimension=4)
        context = make_context(4, [[0, 1], [2, 3]])

        sub_problem = SubSpaceProblem(
            original_problem, context, group_idx=0, dim_indices=[0, 1]
        )
        sub_problem.evaluate([99.0, 99.0])

        assert context.get_full_vector() == [0.0, 0.0, 0.0, 0.0]

class TestBeginIteration:

    def test_begin_iteration_does_not_forward_to_the_original_problem(self):

        original_problem = make_temp_problem(dimension=4)
        context = make_context(4, [[0, 1], [2, 3]])

        sub_problem = SubSpaceProblem(
            original_problem, context, group_idx=0, dim_indices=[0, 1]
        )
        sub_problem.begin_iteration()

        original_problem.begin_iteration.assert_not_called()

# --- Helper functions ---

def make_temp_problem(dimension=6, lower=-5.0, upper=5.0):

    mock_problem = MagicMock(spec=Problem)
    mock_problem.dimension = dimension
    mock_problem.name = "MockProblem"
    mock_problem.bounds = (
        [lower] * dimension,
        [upper] * dimension,
    )

    mock_problem.evaluate.return_value = Evaluation(fitness=0.0)
    mock_problem.is_dynamic.return_value = (False, False)
    mock_problem.is_multi_objective.return_value = False
    return mock_problem


def make_context(dimension, groups):
    """Return a ContextVector initialised to all zeros."""
    return ContextVector(dimension=dimension, groups=groups)
