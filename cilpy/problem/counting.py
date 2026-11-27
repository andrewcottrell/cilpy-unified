# cilpy/problem/counting.py
"""A problem wrapper that counts fitness evaluations."""

from contextlib import contextmanager
from typing import Tuple
from cilpy.problem import Evaluation, Problem

class CountingProblem(Problem):
    """Wraps a problem and counts every call to evaluate.

    The runner uses it to enforce an evaluation budget, and the cooperative
    solvers use it to tell the triggers how many evaluations were used.
    Everything except evaluate is passed straight to the wrapped problem.
    """

    def __init__(self, problem: Problem) -> None:
        """Initializes the counter at zero evaluations.

        Args:
            problem: The problem to wrap.
        """

        self.problem = problem
        self.evaluations = 0
        self._counting = True

    @property
    def name(self) -> str:
        """The name of the wrapped problem."""

        return self.problem.name

    @property
    def dimension(self) -> int:
        """The dimension of the wrapped problem."""

        return self.problem.dimension

    @property
    def bounds(self):
        """The bounds of the wrapped problem."""

        return self.problem.bounds

    def evaluate(self, solution) -> Evaluation:
        """Evaluates the solution and adds one to the count if counting is on.

        Args:
            solution: The solution to evaluate.

        Returns:
            The evaluation from the wrapped problem.
        """
        if self._counting:
            self.evaluations += 1

        return self.problem.evaluate(solution)

    @contextmanager
    def not_counting(self):
        """Turns counting off inside a with block.

        Used for measurements that must not use up the budget. Counting
        goes back to how it was when the block ends.
        """

        was_counting = self._counting
        self._counting = False

        try:
            yield

        finally:
            self._counting = was_counting

    @contextmanager
    def counting(self):
        """Turns counting on inside a with block.

        The runner counts only the solver's own work this way. Counting goes
        back to how it was when the block ends.
        """

        was_counting = self._counting
        self._counting = True

        try:
            yield

        finally:
            self._counting = was_counting

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Returns whether the wrapped problem is dynamic."""

        return self.problem.is_dynamic()

    def is_multi_objective(self) -> bool:
        """Returns whether the wrapped problem is multi-objective."""

        return self.problem.is_multi_objective()

    def begin_iteration(self) -> None:
        """Passes the begin_iteration call to the wrapped problem."""

        self.problem.begin_iteration()

    def get_fitness_bounds(self):
        """Returns the fitness bounds of the wrapped problem."""
        return self.problem.get_fitness_bounds()

    def __getattr__(self, attribute_name):
        """Passes any other attribute lookup to the wrapped problem.

        The problem attribute itself raises an AttributeError, so a lookup
        before __init__ has run does not loop forever.
        """

        if attribute_name == "problem":
            raise AttributeError(attribute_name)


        return getattr(self.problem, attribute_name)
