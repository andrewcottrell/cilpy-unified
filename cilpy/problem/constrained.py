# cilpy/problem/constrained.py
"""
Constrained Benchmark Optimization Problems.

This module provides implementations of common benchmark functions
for single-objective, constrained optimization, adhering to the `Problem`
interface.
"""
from typing import List, Tuple

from numpy import inf

from cilpy.problem import Problem, Evaluation


class G01(Problem[List[float], float]):
    """g01 from the CEC 2006 benchmark suite.

    This is a 13-dimensional minimization problem with nine linear inequality
    constraints.

    The objective function is:
    f(x) = sum_{j=1 to 4}(5*x_j - 5*x_j^2) - sum_{j=5 to 13}(x_j)

    Subject to 9 linear inequality constraints.

    The known global optimum is at
    x* = (1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1) with f(x*) = -15.
    """

    def __init__(self):
        """Initializes a G01 instance."""
        # Bounds: x_j in [0,1] for j=1..9
        #         x_j in [0,100] for j=10..12
        #         x_13 in [0,1]
        lower_bounds = [0.0] * 13
        upper_bounds = [1.0] * 9 + [100.0] * 3 + [1.0]
        super().__init__(dimension=13, bounds=(lower_bounds, upper_bounds), name="G01")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the function for a given solution.

        Args:
            solution: A list of 13 floats.

        Returns:
            An Evaluation object containing the fitness and constraint
            violations.
        """
        x = [s for s in solution]  # Use a copy

        # Objective function
        sum1 = sum(5 * x[j] for j in range(4))
        sum2 = sum(5 * x[j] ** 2 for j in range(4))
        sum3 = sum(x[j] for j in range(4, 13))
        fitness = sum1 - sum2 - sum3

        # Inequality constraints (g(x) <= 0)
        constraints = [
            2 * x[0] + 2 * x[1] + x[9] + x[10] - 10,
            2 * x[0] + 2 * x[2] + x[9] + x[11] - 10,
            2 * x[1] + 2 * x[2] + x[10] + x[11] - 10,
            -8 * x[0] + x[9],
            -8 * x[1] + x[10],
            -8 * x[2] + x[11],
            -2 * x[3] - x[4] + x[9],
            -2 * x[5] - x[6] + x[10],
            -2 * x[7] - x[8] + x[11],
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that this function is not dynamic."""
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        return (-15.0, 5.0)


# ==============================================================================
#  These problems are drawn from Appendix A.6 of Computational Intelligence: An
#  Introduction (second edition) by Andries P. Engelbrecht
# ==============================================================================


class C01(Problem[List[float], float]):
    """The global optimum is x = (0.5, 0.25), with f(x) = 0.25."""

    def __init__(self):
        """Initializes a Problem instance."""
        lower_bounds = [-0.5, -4]
        upper_bounds = [0.5, 1.0]
        super().__init__(dimension=2, bounds=(lower_bounds, upper_bounds), name="C01")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates a candidate solution."""
        x = [s for s in solution]  # Use a copy

        # Objective function
        fitness = 100 * (x[1] - x[0] ** 2) + (1 - x[0]) ** 2

        # Inequality constraints (g(x) <= 0)
        constraints = [
            -x[0] - x[1] ** 2,
            -x[0] ** 2 - x[1],
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that this function is not dynamic."""
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class C02(Problem[List[float], float]):
    """The global optimum is x = (1, 1), with f(x) = 1."""

    def __init__(self):
        """Initializes a Problem instance."""
        lower_bounds = [-2.0, -2.0]
        upper_bounds = [2.0, 2.0]
        super().__init__(dimension=2, bounds=(lower_bounds, upper_bounds), name="C02")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates a candidate solution."""
        x = [s for s in solution]  # Use a copy

        # Objective function
        fitness = (x[0] - 2) ** 2 - (x[1] - 1) ** 2

        # Inequality constraints (g(x) <= 0)
        constraints = [
            -x[0] ** 2 - x[1],
            -x[0] - x[1] - 2,
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that this function is not dynamic."""
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class ConstrainedSphere(Problem[List[float], float]):
    """Sphere function for testing with large dimensions"""

    def __init__(
        self,
        dimension: int,
        domain: Tuple[float, float] = (-100.0, 100.0),
        constraint_rhs: float = None,
    ):
        bounds = ([domain[0]] * dimension, [domain[1]] * dimension)
        super().__init__(dimension, bounds, f"ConstrainedSphere_{dimension}D")
        if constraint_rhs is None:
            self._constraint_rhs = float(-dimension)
        else:
            self._constraint_rhs = float(constraint_rhs)

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        fitness = sum(x ** 2 for x in solution)
        constraint_violation = sum(solution) - self._constraint_rhs
        return Evaluation(fitness=fitness, constraints_inequality=[constraint_violation])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        lower_bound = abs(self.bounds[0][0])
        upper_bound = abs(self.bounds[1][0])
        largest_bound = max(lower_bound, upper_bound)
        minimum_fitness = 0.0
        maximum_fitness = float(self.dimension) * largest_bound ** 2
        return (minimum_fitness, maximum_fitness)
