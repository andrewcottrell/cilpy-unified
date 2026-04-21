# cilpy/solver/decomposition/sub_problem.py
from typing import List, Tuple

from cilpy.problem import Problem, Evaluation
from cilpy.solver.decomposition.context import ContextVector


class SubSpaceProblem(Problem):
    """
    A problem that contains only a sub-set of the original dimensions.

    Each sub-swarm in a cooperative decomposition solver works on its own
    piece of the decision variables. This class wraps the original problem
    and presents only the dimensions assigned to one group. When a sub-swarm
    evaluates a candidate partial solution, this class fills the remaining
    dimensions from the shared context vector and then calls the original
    problem's evaluate method, so the full-dimensional fitness are returned unchanged.
    """

    def __init__(
        self,
        original_problem: Problem,
        context: ContextVector,
        group_idx: int,
        dim_indices: List[int],
        name: str = "",
    ) -> None:
        """
        Initializes the SubSpaceProblem.

        Args:
            original_problem: The full-dimensional problem being wrapped.
            context: The shared context vector that holds the current best
                values for all dimensions across all groups.
            group_idx: The index of this group inside the context vector.
            dim_indices: The list of dimension indices owned by this
                sub-problem. Its length becomes the sub-problem's dimension.
            name: Optional name for this sub problem.
        """
        lower_bounds = []
        upper_bounds = []
        for global_idx in dim_indices:
            lower_bounds.append(original_problem.bounds[0][global_idx])
            upper_bounds.append(original_problem.bounds[1][global_idx])

        sub_bounds = (lower_bounds, upper_bounds)

        sub_name = name if name else f"{original_problem.name}_sub{group_idx}"

        super().__init__(len(dim_indices), sub_bounds, sub_name)

        self._original_problem = original_problem
        self._context = context
        self._group_idx = group_idx
        self._dim_indices = list(dim_indices)

    def evaluate(self, sub_solution: List[float]) -> Evaluation:
        """
        Evaluates a partial solution in the context of the full problem.

        The sub-solution is inserted into a copy of the full context vector
        at the positions owned by this group. The original problem is then
        evaluated on the full vector and the result is returned,
        including any constraint values.

        Args:
            sub_solution: A candidate solution for the dimensions owned by
                this group. Must have the same length as dim_indices.

        Returns:
            The Evaluation returned by the original problem for the
            full-dimensional solution.
        """
        full_vector = self._context.get_full_vector()

        for k, global_idx in enumerate(self._dim_indices):

            full_vector[global_idx] = sub_solution[k]

        return self._original_problem.evaluate(full_vector)

    def update_dim_indices(self, new_indices: List[int]) -> None:
        """
        Updates the group's dimension indices after a regroup event happend.

        Called when variables are reassigned between
        groups. Updates the sub-problem's dimension count and bounds in
        place so the same object can be reused with a new sub-solver.

        Args:
            new_indices: The new list of global dimension indices for this
                group.
        """
        self._dim_indices = list(new_indices)
        self.dimension = len(new_indices)

        lower_bounds = []
        upper_bounds = []

        for global_idx in new_indices:
            lower_bounds.append(self._original_problem.bounds[0][global_idx])
            upper_bounds.append(self._original_problem.bounds[1][global_idx])

        self.bounds = (lower_bounds, upper_bounds)

    # Methods for original problem

    def begin_iteration(self) -> None:
        """
        A notification called by the ExperimentRunner before each solver
        iteration (just like in the original problem).
        """
        pass

    def is_dynamic(self) -> Tuple[bool, bool]:
        """
        Returns true if the original problem is dynamic in its objective or constraints.

        Returns:
            A tuple (is_objective_dynamic, is_constraint_dynamic).
        """
        return self._original_problem.is_dynamic()

    def is_multi_objective(self) -> bool:
        """
        Returns true if the original problem has multiple objectives.

        Returns:
            True if the original problem has multiple objectives.
        """
        return self._original_problem.is_multi_objective()

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """
        Returns the fitness bounds of the original problem.

        Returns:
            A tuple (global_minimum_fitness, global_maximum_fitness).
        """
        return self._original_problem.get_fitness_bounds()

