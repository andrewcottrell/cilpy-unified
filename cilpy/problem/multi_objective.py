# cilpy/problem/multi_objective.py
"""Static multi-objective benchmark problems.

Box-constrained: SCH1 (Schaffer, 1985) and ZDT1-4, ZDT6 (Zitzler, Deb and
Thiele, 2000). Constrained: BNH, SRN, TNK, OSY and CONSTR (Deb, 2001; Deb
et al., 2002). Constraints use g(x) <= 0.

true_pareto_front(n_points) returns the analytic front for SCH1 and ZDT, and
a grid-sampled front for the two-variable constrained problems. OSY has none.
"""

import math
from typing import List, Tuple

import numpy as np

from cilpy.problem import Problem, Evaluation


def _nondominated(objectives: np.ndarray) -> np.ndarray:
    """Filters an (n, nm) array of objective vectors to the non-dominated set."""
    n = objectives.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        others = np.delete(objectives, i, axis=0)
        dominated = np.any(
            np.all(others <= objectives[i], axis=1)
            & np.any(others < objectives[i], axis=1)
        )
        if dominated:
            keep[i] = False
    return objectives[keep]


class SCH1(Problem[List[float], List[float]]):
    """The Schaffer Function N. 1.

    Single-variable, two-objective problem with a convex front.

        f1(x) = x^2
        f2(x) = (x - 2)^2

    The Pareto-optimal set is x in [0, 2].

    Reference: [Schaffer, 1985]; also Deb (2001), Section 8.3.
    """

    def __init__(self, domain: Tuple[float, float] = (-10.0, 10.0)):
        """Initializes a Schaffer Function N. 1 problem instance.

        Args:
            domain: A tuple `(min_val, max_val)` bounding the single decision
                variable. Defaults to (-10.0, 10.0).
        """
        super().__init__(dimension=1, bounds=([domain[0]], [domain[1]]), name="SCH1")

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = solution[0]
        return Evaluation(fitness=[x**2, (x - 2) ** 2])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return True

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        x = np.linspace(0.0, 2.0, n_points)
        return np.column_stack([x**2, (x - 2) ** 2])


# ZDT suite: f2 = g(x2..xn) h(f1, g); the front is at g = 1.


class _ZDTBase(Problem[List[float], List[float]]):
    """Shared plumbing for the ZDT family."""

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return True


class ZDT1(_ZDTBase):
    """ZDT1: convex Pareto-optimal front.

        f1(x) = x1
        g(x)  = 1 + 9 * sum(x2..xn) / (n - 1)
        f2(x) = g * (1 - sqrt(f1 / g))

    with x in [0, 1]^n and n = 30 by default. The POF is
    f2 = 1 - sqrt(f1) for f1 in [0, 1], attained at g = 1.

    Reference: [Zitzler et al., 2000], problem T1.
    """

    def __init__(self, dimension: int = 30):
        super().__init__(
            dimension=dimension,
            bounds=([0.0] * dimension, [1.0] * dimension),
            name="ZDT1",
        )

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.dimension - 1)
        f2 = g * (1.0 - math.sqrt(f1 / g))
        return Evaluation(fitness=[float(f1), float(f2)])

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT2(_ZDTBase):
    """ZDT2: concave (non-convex) Pareto-optimal front.

        f2(x) = g * (1 - (f1 / g)^2)

    otherwise as ZDT1. The POF is f2 = 1 - f1^2 for f1 in [0, 1].

    Reference: [Zitzler et al., 2000], problem T2.
    """

    def __init__(self, dimension: int = 30):
        super().__init__(
            dimension=dimension,
            bounds=([0.0] * dimension, [1.0] * dimension),
            name="ZDT2",
        )

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.dimension - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return Evaluation(fitness=[float(f1), float(f2)])

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - f1**2])


class ZDT3(_ZDTBase):
    """ZDT3: disconnected Pareto-optimal front.

        f2(x) = g * (1 - sqrt(f1/g) - (f1/g) * sin(10 * pi * f1))

    otherwise as ZDT1. The sine term splits the POF into five disconnected
    convex segments.

    Reference: [Zitzler et al., 2000], problem T3.
    """

    def __init__(self, dimension: int = 30):
        super().__init__(
            dimension=dimension,
            bounds=([0.0] * dimension, [1.0] * dimension),
            name="ZDT3",
        )

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.dimension - 1)
        ratio = f1 / g
        f2 = g * (1.0 - math.sqrt(ratio) - ratio * math.sin(10.0 * math.pi * f1))
        return Evaluation(fitness=[float(f1), float(f2)])

    def true_pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        # The g=1 surface is only Pareto-optimal on the non-dominated pieces.
        return _nondominated(np.column_stack([f1, f2]))


class ZDT4(_ZDTBase):
    """ZDT4: highly multimodal problem with 21^9 local Pareto fronts.

        g(x) = 1 + 10(n-1) + sum_{i=2..n} (xi^2 - 10 cos(4 pi xi))

    with x1 in [0, 1], xi in [-5, 5] for i >= 2, and n = 10 by default. The
    global POF equals ZDT1's (f2 = 1 - sqrt(f1)); the Rastrigin-like g
    creates 21^9 local fronts.

    Reference: [Zitzler et al., 2000], problem T4.
    """

    def __init__(self, dimension: int = 10):
        lower = [0.0] + [-5.0] * (dimension - 1)
        upper = [1.0] + [5.0] * (dimension - 1)
        super().__init__(dimension=dimension, bounds=(lower, upper), name="ZDT4")

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        f1 = x[0]
        tail = x[1:]
        g = (
            1.0
            + 10.0 * (self.dimension - 1)
            + np.sum(tail**2 - 10.0 * np.cos(4.0 * np.pi * tail))
        )
        f2 = g * (1.0 - math.sqrt(f1 / g))
        return Evaluation(fitness=[float(f1), float(f2)])

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT6(_ZDTBase):
    """ZDT6: non-uniform solution density and a concave front.

        f1(x) = 1 - exp(-4 x1) * sin^6(6 pi x1)
        g(x)  = 1 + 9 * (sum(x2..xn) / (n - 1))^0.25
        f2(x) = g * (1 - (f1 / g)^2)

    with x in [0, 1]^n and n = 10 by default. Solution density is lowest near
    the front and non-uniform along it.

    Reference: [Zitzler et al., 2000], problem T6.
    """

    def __init__(self, dimension: int = 10):
        super().__init__(
            dimension=dimension,
            bounds=([0.0] * dimension, [1.0] * dimension),
            name="ZDT6",
        )

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        f1 = 1.0 - math.exp(-4.0 * x[0]) * math.sin(6.0 * math.pi * x[0]) ** 6
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (self.dimension - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return Evaluation(fitness=[float(f1), float(f2)])

    def true_pareto_front(self, n_points: int = 1000) -> np.ndarray:
        x1 = np.linspace(0.0, 1.0, n_points)
        f1 = 1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1) ** 6
        f2 = 1.0 - f1**2
        return _nondominated(np.column_stack([f1, f2]))


# Constrained problems.


class _SampledFrontMixin:
    """Grid-sampled reference front for two-variable problems. Subclasses
    provide `_batch_evaluate(X1, X2) -> (F, G)`. Cached after the first call.
    """

    _GRID = 1000

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        if getattr(self, "_cached_front", None) is None:
            lower, upper = self.bounds
            a = np.linspace(lower[0], upper[0], self._GRID)
            b = np.linspace(lower[1], upper[1], self._GRID)
            X1, X2 = np.meshgrid(a, b)
            X1, X2 = X1.ravel(), X2.ravel()

            F, G = self._batch_evaluate(X1, X2)
            feasible = np.all(G <= 1e-9, axis=1)
            objectives = F[feasible]

            # Fast ND filter for bi-objective sets: sort by f1 (ties by f2)
            # and sweep, keeping points with strictly decreasing f2.
            order = np.lexsort((objectives[:, 1], objectives[:, 0]))
            objectives = objectives[order]
            best_f2 = np.inf
            keep = np.zeros(len(objectives), dtype=bool)
            for i, (_, f2) in enumerate(objectives):
                if f2 < best_f2:
                    keep[i] = True
                    best_f2 = f2
            self._cached_front = objectives[keep]

        front = self._cached_front
        if len(front) > n_points:
            idx = np.linspace(0, len(front) - 1, n_points).astype(int)
            front = front[idx]
        return front.copy()


class BNH(_SampledFrontMixin, Problem[List[float], List[float]]):
    """Binh and Korn's constrained bi-objective problem.

        min f1(x) = 4 x1^2 + 4 x2^2
        min f2(x) = (x1 - 5)^2 + (x2 - 5)^2
        s.t. (x1 - 5)^2 + x2^2 <= 25
             (x1 - 8)^2 + (x2 + 3)^2 >= 7.7

    with x1 in [0, 5], x2 in [0, 3]. The constraints are inactive on the
    Pareto front.

    Reference: [Binh & Korn, 1997]; also Deb et al. (2002).
    """

    def __init__(self):
        super().__init__(dimension=2, bounds=([0.0, 0.0], [5.0, 3.0]), name="BNH")

    def _batch_evaluate(self, X1, X2):
        F = np.column_stack([
            4.0 * X1**2 + 4.0 * X2**2,
            (X1 - 5.0) ** 2 + (X2 - 5.0) ** 2,
        ])
        G = np.column_stack([
            (X1 - 5.0) ** 2 + X2**2 - 25.0,
            7.7 - (X1 - 8.0) ** 2 - (X2 + 3.0) ** 2,
        ])
        return F, G

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = solution
        f1 = 4.0 * x1**2 + 4.0 * x2**2
        f2 = (x1 - 5.0) ** 2 + (x2 - 5.0) ** 2
        g1 = (x1 - 5.0) ** 2 + x2**2 - 25.0
        g2 = 7.7 - (x1 - 8.0) ** 2 - (x2 + 3.0) ** 2
        return Evaluation(fitness=[f1, f2], constraints_inequality=[g1, g2])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return True


class SRN(_SampledFrontMixin, Problem[List[float], List[float]]):
    """Srinivas and Deb's constrained bi-objective problem.

        min f1(x) = 2 + (x1 - 2)^2 + (x2 - 1)^2
        min f2(x) = 9 x1 - (x2 - 1)^2
        s.t. x1^2 + x2^2 <= 225
             x1 - 3 x2 + 10 <= 0

    with x1, x2 in [-20, 20]. The constraints eliminate part of the
    unconstrained Pareto front, so the constrained front differs from the
    unconstrained one.

    Reference: [Srinivas & Deb, 1994]; also Deb et al. (2002).
    """

    def __init__(self):
        super().__init__(
            dimension=2, bounds=([-20.0, -20.0], [20.0, 20.0]), name="SRN"
        )

    def _batch_evaluate(self, X1, X2):
        F = np.column_stack([
            2.0 + (X1 - 2.0) ** 2 + (X2 - 1.0) ** 2,
            9.0 * X1 - (X2 - 1.0) ** 2,
        ])
        G = np.column_stack([
            X1**2 + X2**2 - 225.0,
            X1 - 3.0 * X2 + 10.0,
        ])
        return F, G

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = solution
        f1 = 2.0 + (x1 - 2.0) ** 2 + (x2 - 1.0) ** 2
        f2 = 9.0 * x1 - (x2 - 1.0) ** 2
        g1 = x1**2 + x2**2 - 225.0
        g2 = x1 - 3.0 * x2 + 10.0
        return Evaluation(fitness=[f1, f2], constraints_inequality=[g1, g2])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return True


class TNK(_SampledFrontMixin, Problem[List[float], List[float]]):
    """Tanaka's constrained bi-objective problem.

        min f1(x) = x1
        min f2(x) = x2
        s.t. x1^2 + x2^2 - 1 - 0.1 cos(16 arctan(x1 / x2)) >= 0
             (x1 - 0.5)^2 + (x2 - 0.5)^2 <= 0.5

    with x1, x2 in [0, pi]. The Pareto front lies on the first constraint
    boundary and is disconnected.

    Reference: [Tanaka et al., 1995]; also Deb et al. (2002).
    """

    def __init__(self):
        super().__init__(
            dimension=2, bounds=([0.0, 0.0], [math.pi, math.pi]), name="TNK"
        )

    def _batch_evaluate(self, X1, X2):
        F = np.column_stack([X1, X2])
        angle = np.arctan2(X1, X2)
        G = np.column_stack([
            -(X1**2 + X2**2 - 1.0 - 0.1 * np.cos(16.0 * angle)),
            (X1 - 0.5) ** 2 + (X2 - 0.5) ** 2 - 0.5,
        ])
        return F, G

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = solution
        f1, f2 = x1, x2
        # arctan(x1/x2) with the conventional limit at x2 = 0.
        angle = math.atan2(x1, x2)
        g1 = -(x1**2 + x2**2 - 1.0 - 0.1 * math.cos(16.0 * angle))
        g2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 - 0.5
        return Evaluation(fitness=[f1, f2], constraints_inequality=[g1, g2])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return True


class CONSTR(_SampledFrontMixin, Problem[List[float], List[float]]):
    """Deb's CONSTR problem.

        min f1(x) = x1
        min f2(x) = (1 + x2) / x1
        s.t. x2 + 9 x1 >= 6
             -x2 + 9 x1 >= 1

    with x1 in [0.1, 1], x2 in [0, 5]. Part of the constrained front lies on
    a constraint boundary and part on the unconstrained front.

    Reference: Deb (2001), Section 7.2; also Deb et al. (2002).
    """

    def __init__(self):
        super().__init__(dimension=2, bounds=([0.1, 0.0], [1.0, 5.0]), name="CONSTR")

    def _batch_evaluate(self, X1, X2):
        F = np.column_stack([X1, (1.0 + X2) / X1])
        G = np.column_stack([
            6.0 - X2 - 9.0 * X1,
            1.0 + X2 - 9.0 * X1,
        ])
        return F, G

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = solution
        f1 = x1
        f2 = (1.0 + x2) / x1
        g1 = 6.0 - x2 - 9.0 * x1
        g2 = 1.0 + x2 - 9.0 * x1
        return Evaluation(fitness=[f1, f2], constraints_inequality=[g1, g2])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return True


class OSY(Problem[List[float], List[float]]):
    """Osyczka and Kundu's constrained bi-objective problem.

        min f1(x) = -(25(x1-2)^2 + (x2-2)^2 + (x3-1)^2 + (x4-4)^2 + (x5-1)^2)
        min f2(x) = x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2
        s.t. x1 + x2 - 2 >= 0
             6 - x1 - x2 >= 0
             2 - x2 + x1 >= 0
             2 - x1 + 3 x2 >= 0
             4 - (x3 - 3)^2 - x4 >= 0
             (x5 - 3)^2 + x6 - 4 >= 0

    with x1, x2, x6 in [0, 10], x3, x5 in [1, 5], x4 in [0, 6]. Six decision
    variables and six constraints; the Pareto front is a union of five
    segments, each lying on the intersection of active constraint
    boundaries. `true_pareto_front` is not implemented.

    Reference: [Osyczka & Kundu, 1995]; also Deb et al. (2002).
    """

    def __init__(self):
        lower = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        upper = [10.0, 10.0, 5.0, 6.0, 5.0, 10.0]
        super().__init__(dimension=6, bounds=(lower, upper), name="OSY")

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2, x3, x4, x5, x6 = solution
        f1 = -(
            25.0 * (x1 - 2.0) ** 2
            + (x2 - 2.0) ** 2
            + (x3 - 1.0) ** 2
            + (x4 - 4.0) ** 2
            + (x5 - 1.0) ** 2
        )
        f2 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
        constraints = [
            -(x1 + x2 - 2.0),
            -(6.0 - x1 - x2),
            -(2.0 - x2 + x1),
            -(2.0 - x1 + 3.0 * x2),
            -(4.0 - (x3 - 3.0) ** 2 - x4),
            -((x5 - 3.0) ** 2 + x6 - 4.0),
        ]
        return Evaluation(fitness=[f1, f2], constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return True
