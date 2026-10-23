# cilpy/problem/unconstrained.py
"""
Benchmark Optimization Functions.

This module provides concrete implementations of common benchmark functions
for single-objective, unconstrained optimization, adhering to the `Problem`
interface.
"""
import math
from typing import List, Tuple

from cilpy.problem import Problem, Evaluation


class Sphere(Problem[List[float], float]):
    """The Sphere function.

    A continuous, convex, and unimodal benchmark function. It is one of the
    simplest benchmark problems. The global minimum is at the origin (0, ..., 0)
    with a fitness value of 0.

    The function is defined as: f(x) = Sum(x_i^2) for i = 1 to n.
    """

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-100, 100)):
        """Initializes a Sphere function problem instance.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-100, 100).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Sphere"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Sphere function for a given solution.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness
                (a single float) and no constraint violations.
        """
        fitness = sum(x**2 for x in solution)
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Sphere function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """
        Returns the known theoretical min and max fitness values for the
        problem.

        This is used for calculating normalized performance metrics.

        Returns:
            A tuple containing (global_minimum_fitness, global_maximum_fitness).
        """
        # --- Best fitness ---
        best_fitness = 0.0

        # --- Worst fitness ---
        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]
        limit = max(abs(lower_bound), abs(upper_bound))
        worst_fitness = self.dimension * (limit**2)

        return (best_fitness, worst_fitness)


class Quadratic(Problem[List[float], float]):
    """The Quadratic function.

    Also known as the Schwefel 1.2 or Double Sum function. It is a continuous,
    convex, and unimodal benchmark function. It is more difficult than the
    Sphere  function because the variables are dependent on each other due to
    the cumulative summation.

    The global minimum is at the origin (0, ..., 0) with a fitness value of 0.

    The function is defined as: f(x) = Sum_{j=1}^n ( Sum_{k=1}^j x_k )^2
    """

    def __init__(
        self, dimension: int = 2, domain: Tuple[float, float] = (-100.0, 100.0)
    ):
        """Initializes a Quadric function problem instance.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-100, 100).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Quadric"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Quadric function for a given solution.

        Implements f(x) = Σ(Σ(x_k)^2) efficiently using a running sum.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness.
        """
        fitness = 0.0
        cumulative_sum = 0.0
        for x in solution:
            cumulative_sum += x
            fitness += cumulative_sum**2

        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Quadric function is not dynamic.

        Returns:
            Tuple[bool, bool]: (False, False).
        """
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        best_fitness = 0.0

        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]
        limit = max(abs(lower_bound), abs(upper_bound))

        n = self.dimension
        sum_of_squares = n * (n + 1) * (2 * n + 1) / 6
        worst_fitness = (limit**2) * sum_of_squares

        return (best_fitness, worst_fitness)


class Ackley(Problem[List[float], float]):
    """The Ackley function.

    A widely used multimodal benchmark function characterized by a nearly flat
    outer region and a large number of local minima. Its global minimum is at
    the origin (0, ..., 0) with a fitness value of 0.

    The function is defined as:
    f(x) = -a * exp(-b * sqrt(1/n * Sum(x_i^2)))
          - exp(1/n * Sum(cos(c*x_i)))
          + a
          + exp(1)
    """

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-32, 32)):
        """Initializes an Ackley function problem instance.

        The standard parameters `a=20`, `b=0.2`, and `c=2pi` are used.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-32, 32).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Ackley"
        )
        # Standard parameters for the Ackley function
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Ackley function for a given solution.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness
                (a single float) and no constraint violations.
        """
        sum_sq = sum(x**2 for x in solution)
        sum_cos = sum(math.cos(self.c * x) for x in solution)

        term1 = -self.a * math.exp(-self.b * math.sqrt(sum_sq / self.dimension))
        term2 = -math.exp(sum_cos / self.dimension)

        fitness = term1 + term2 + self.a + math.e
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Ackley function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:

        best_fitness = 0.0
        worst_fitness = self.a + math.e

        return (best_fitness, worst_fitness)


class Rastrigin(Problem[List[float], float]):
    """The Rastrigin function.

    A highly multimodal benchmark function with a large number of local
    minima. Its global minimum is at the
    origin (0, ..., 0) with a fitness value of 0.

    The function is defined as:
    f(x) = A*n + Sum(x_i^2 - A*cos(2*pi*x_i)) for i = 1 to n
    """

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-5.12, 5.12)):
        """Initializes a Rastrigin function problem instance.

        The standard parameter `A=10` is used.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-5.12, 5.12).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Rastrigin"
        )
        self.A = 10

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Rastrigin function for a given solution.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness
                (a single float) and no constraint violations.
        """
        per_dimension_sum = sum(
            x**2 - self.A * math.cos(2 * math.pi * x) for x in solution
        )
        fitness = self.A * self.dimension + per_dimension_sum
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Rastrigin function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """Returns the known theoretical min and max fitness values.

        Returns:
            A tuple containing (global_minimum_fitness, global_maximum_fitness).
        """
        best_fitness = 0.0

        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]
        limit = max(abs(lower_bound), abs(upper_bound))

        worst_per_dimension = limit**2 + self.A
        worst_fitness = self.dimension * worst_per_dimension

        return (best_fitness, worst_fitness)


class Rosenbrock(Problem[List[float], float]):
    """The generalized Rosenbrock function.

    A widely used multimodal, non-separable benchmark function.
    The global minimum is at (1, ..., 1) with a fitness value of 0.

    The function is defined as:
    f(x) = Sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    """

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-30, 30)):
        """Initializes a generalized Rosenbrock function problem instance.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2. At least two are needed for this function.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-30, 30).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Rosenbrock"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the generalized Rosenbrock function for a given solution.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness
                (a single float) and no constraint violations.
        """
        fitness = 0.0

        for i in range(len(solution) - 1):
            current_value = solution[i]
            next_value = solution[i + 1]

            valley_term = 100.0 * (next_value - current_value**2) ** 2

            offset_term = (current_value - 1) ** 2

            fitness += valley_term + offset_term

        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Rosenbrock function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """Returns the known theoretical min and max fitness values.

        Returns:
            A tuple containing (global_minimum_fitness, global_maximum_fitness).
        """

        best_fitness = 0.0

        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]
        limit = max(abs(lower_bound), abs(upper_bound))

        worst_valley_term = 100.0 * (limit + limit**2) ** 2
        worst_offset_term = (limit + 1) ** 2

        number_of_terms = max(0, self.dimension - 1)
        worst_fitness = number_of_terms * (worst_valley_term + worst_offset_term)

        return (best_fitness, worst_fitness)


class Griewank(Problem[List[float], float]):
    """The Griewank function.

    A multimodal, non-separable benchmark function with many regularly spaced
    local minima. The global minimum is at the origin (0, ..., 0) with a
    fitness value of 0.

    The function is defined as:
    f(x) = Sum(x_i^2) / 4000 - Prod(cos(x_i / sqrt(i))) + 1 for i = 1 to n
    """

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-600, 600)):
        """Initializes a Griewank function problem instance.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-600, 600).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Griewank"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Griewank function for a given solution.

        Args:
            solution: A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation: An Evaluation object containing the fitness
        """

        sum_of_squares = 0.0
        product_of_cosines = 1.0

        for i, x in enumerate(solution, start=1):
            sum_of_squares += x**2
            product_of_cosines *= math.cos(x / math.sqrt(i))

        fitness = sum_of_squares / 4000 - product_of_cosines + 1
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Griewank function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:

        best_fitness = 0.0

        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]

        limit = max(abs(lower_bound), abs(upper_bound))

        worst_fitness = self.dimension * limit**2 / 4000 + 2

        return (best_fitness, worst_fitness)


class Elliptic(Problem[List[float], float]):
    """The high-conditioned Elliptic function"""

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-100, 100)):
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Elliptic"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Elliptic function for a given solution."""
        
        last_index = max(1, len(solution) - 1)
        fitness = 0.0

        for i, x in enumerate(solution):
            weight = (10**6) ** (i / last_index)
            fitness += weight * x**2

        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Griewank function is not dynamic"""
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        best_fitness = 0.0

        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]
        limit = max(abs(lower_bound), abs(upper_bound))

        last_index = max(1, self.dimension - 1)
        total_weight = 0.0
        for i in range(self.dimension):
            total_weight += (10**6) ** (i / last_index)

        worst_fitness = total_weight * limit**2

        return (best_fitness, worst_fitness)


class Schwefel2_22(Problem[List[float], float]):
    """The Schwefel 2.22 function.

    A unimodal and non-separable benchmark function. The global minimum is at the
    origin with a fitness value of 0.

    The function is defined as:
    f(x) =  Sum(|x_i|) + Prod(|x_i|) for i = 1 to n
    """

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-10, 10)):
        """Initializes a Schwefel 2.22 function problem instance.
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension

        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Schwefel2_22")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Schwefel 2.22 function for a given solution."""

        sum_of_absolutes = 0.0
        product_of_absolutes = 1.0

        for x in solution:
            sum_of_absolutes += abs(x)
            product_of_absolutes *= abs(x)

        fitness = sum_of_absolutes + product_of_absolutes
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the function is not dynamic."""
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """Returns the known theoretical min and max fitness values"""
        best_fitness = 0.0

        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]
        limit = float(max(abs(lower_bound), abs(upper_bound)))

        try:
            worst_product = limit**self.dimension
        except OverflowError:
            worst_product = float("inf")

        worst_fitness = self.dimension * limit + worst_product

        return (best_fitness, worst_fitness)


class Schwefel2_26(Problem[List[float], float]):
    """The Schwefel 2.26 function"""

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-500, 500)):
        """Initializes a Schwefel 2.26 function problem instance.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-500, 500).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension

        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Schwefel2_26")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Schwefel 2.26 function for a given solutoin."""
        total = 0.0

        for x in solution:
            total += x * math.sin(math.sqrt(abs(x)))

        fitness = 418.9829 * len(solution) - total

        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the function is not dynamic."""
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """Returns the known theoretical min and max fitness values"""
        best_fitness = 0.0

        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]

        number_of_points = 100001
        step = (upper_bound - lower_bound) / (number_of_points - 1)
        worst_per_dimension = 0.0

        for i in range(number_of_points):
            x = lower_bound + i * step
            value = 418.9829 - x * math.sin(math.sqrt(abs(x)))
            worst_per_dimension = max(worst_per_dimension, value)

        worst_fitness = self.dimension * worst_per_dimension

        return (best_fitness, worst_fitness)



if __name__ == "__main__":
    my_sphere = Sphere(2, (-5.12, 5.12))
    print(my_sphere.evaluate([0, 0]))

    my_quadratic = Quadratic(2, (-5.12, 5.12))
    print(my_quadratic.evaluate([0, 0]))

    my_ackley = Ackley(2, (-5.12, 5.12))
    print(my_ackley.evaluate([0, 0]))

    my_rastrigin = Rastrigin(2, (-5.12, 5.12))
    print(my_rastrigin.evaluate([0, 0]))

    my_rosenbrock = Rosenbrock(2, (-30, 30))
    print(my_rosenbrock.evaluate([1, 1]))

    my_schwefel = Schwefel2_22(2, (-10, 10))
    print(my_schwefel.evaluate([0, 0]))

    my_schwefel_2_26 = Schwefel2_26(2, (-500, 500))
    print(my_schwefel_2_26.evaluate([420.9687, 420.9687]))
