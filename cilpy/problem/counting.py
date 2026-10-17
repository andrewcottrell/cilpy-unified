# cilpy/problem/counting.py
from contextlib import contextmanager
from typing import Tuple
from cilpy.problem import Evaluation, Problem

class CountingProblem(Problem):
    """Wraps a problem and counts every call to evaluate."""

    def __init__(self, problem: Problem) -> None:
        
        self.problem = problem
        self.evaluations = 0
        self._counting = True

    @property
    def name(self) -> str:
        
        return self.problem.name

    @property
    def dimension(self) -> int:
        
        return self.problem.dimension

    @property
    def bounds(self):
        
        return self.problem.bounds

    def evaluate(self, solution) -> Evaluation:
        if self._counting:
            self.evaluations += 1

        return self.problem.evaluate(solution)

    @contextmanager
    def not_counting(self):

        was_counting = self._counting
        self._counting = False

        try:
            yield

        finally:
            self._counting = was_counting

    def is_dynamic(self) -> Tuple[bool, bool]:
        
        return self.problem.is_dynamic()

    def is_multi_objective(self) -> bool:
        
        return self.problem.is_multi_objective()

    def begin_iteration(self) -> None:
        
        self.problem.begin_iteration()

    def get_fitness_bounds(self):
        return self.problem.get_fitness_bounds()

    def __getattr__(self, attribute_name):

        if attribute_name == "problem":
            raise AttributeError(attribute_name)


        return getattr(self.problem, attribute_name)
