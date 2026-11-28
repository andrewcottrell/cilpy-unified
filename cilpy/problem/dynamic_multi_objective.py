# cilpy/problem/dynamic_multi_objective.py
"""Dynamic multi-objective benchmark problems.

Time follows Farina, Deb and Amato (2004): begin_iteration() advances an
iteration counter tau, and t = floor(tau / tau_t) / n_t, where tau_t is the
change frequency and n_t the change severity.

- FDA1, FDA3: dynamic objectives, box constraints only (Farina et al., 2004).
- DTNK: static objectives, dynamic constraints (SODC).
- DTNK5: dynamic objectives, static constraints (DOSC).
- DTNK6, DTNK4: dynamic objectives and constraints (DODC).

The DTNK problems take a `dimension` (default 2). Variables x3..xn scale
the first two by g = 1 + sum_{i>=3} (xi - pi/2)^2, and the 2-D problem is
evaluated at (g x1, g x2). The Pareto set has xi = pi/2 for i >= 3, so the
front is that of the 2-D problem.

true_pareto_front() returns the front of the current environment.
"""

import math
from typing import List, Tuple

import numpy as np

from cilpy.problem import Problem, Evaluation


class _DynamicMOBase(Problem[List[float], List[float]]):
    """Shared time mechanics for dynamic multi-objective problems."""

    def __init__(self, dimension, bounds, name, tau_t: int = 10, n_t: int = 10):
        """Initializes the time state.

        Args:
            dimension, bounds, name: As for Problem.
            tau_t: Change frequency: iterations per environment.
            n_t: Change severity; t advances by 1/n_t per change.
        """
        super().__init__(dimension=dimension, bounds=bounds, name=name)
        self.tau_t = tau_t
        self.n_t = n_t
        self._tau = 0

    @property
    def t(self) -> float:
        """Current environment time."""
        return (self._tau // self.tau_t) / self.n_t

    def begin_iteration(self) -> None:
        """Advances the iteration counter (called by the ExperimentRunner)."""
        self._tau += 1

    @property
    def environment(self) -> int:
        """Current environment index k = floor(tau / tau_t)."""
        return self._tau // self.tau_t

    def set_environment(self, k: int) -> None:
        """Sets the current environment index k (t = k / n_t)."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self._tau = k * self.tau_t

    def reset_time(self) -> None:
        """Resets time to zero (call between independent runs)."""
        self._tau = 0

    def is_multi_objective(self) -> bool:
        return True


class FDA1(_DynamicMOBase):
    """FDA1: moving Pareto set, time-invariant Pareto front.

        f1(x)   = x1
        g(x, t) = 1 + sum_{i=2..n} (xi - G(t))^2,   G(t) = sin(0.5 pi t)
        f2(x)   = g * (1 - sqrt(f1 / g))

    with x1 in [0, 1], xi in [-1, 1] for i >= 2 (n = 20 by default). The
    Pareto-optimal set is xi = G(t) for i >= 2, which sweeps through the
    decision space, while the front remains f2 = 1 - sqrt(f1).

    Reference: [Farina et al., 2004], problem FDA1.
    """

    def __init__(self, dimension: int = 20, tau_t: int = 10, n_t: int = 10):
        lower = [0.0] + [-1.0] * (dimension - 1)
        upper = [1.0] + [1.0] * (dimension - 1)
        super().__init__(dimension, (lower, upper), "FDA1", tau_t, n_t)

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        G = math.sin(0.5 * math.pi * self.t)
        f1 = x[0]
        g = 1.0 + float(np.sum((x[1:] - G) ** 2))
        f2 = g * (1.0 - math.sqrt(f1 / g))
        return Evaluation(fitness=[float(f1), float(f2)])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, False)

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class FDA3(_DynamicMOBase):
    """FDA3: moving Pareto front with time-varying solution density.

        f1(x, t) = x1^F(t),                    F(t) = 10^(2 sin(0.5 pi t))
        g(x, t)  = 1 + G(t) + sum_{i=2..n} (xi - G(t))^2,
                                               G(t) = |sin(0.5 pi t)|
        f2(x, t) = g * (1 - sqrt(f1 / g))

    with x1 in [0, 1], xi in [-1, 1] for i >= 2 (n = 20 by default). The
    front f2 = (1 + G(t)) (1 - sqrt(f1 / (1 + G(t)))) moves vertically with
    G(t), and F(t) redistributes solution density along f1.

    Reference: [Farina et al., 2004], problem FDA3.
    """

    def __init__(self, dimension: int = 20, tau_t: int = 10, n_t: int = 10):
        lower = [0.0] + [-1.0] * (dimension - 1)
        upper = [1.0] + [1.0] * (dimension - 1)
        super().__init__(dimension, (lower, upper), "FDA3", tau_t, n_t)

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        G = abs(math.sin(0.5 * math.pi * self.t))
        F = 10.0 ** (2.0 * math.sin(0.5 * math.pi * self.t))
        f1 = float(x[0]) ** F
        g = 1.0 + G + float(np.sum((x[1:] - G) ** 2))
        f2 = g * (1.0 - math.sqrt(f1 / g))
        return Evaluation(fitness=[float(f1), float(f2)])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, False)

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        G = abs(math.sin(0.5 * math.pi * self.t))
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = (1.0 + G) * (1.0 - np.sqrt(f1 / (1.0 + G)))
        return np.column_stack([f1, f2])


def _dtnk_bounds(dimension: int):
    """Bounds [0, pi]^n of a DTNK problem."""
    if dimension < 2:
        raise ValueError(f"dimension must be at least 2, got {dimension}")
    return ([0.0] * dimension, [math.pi] * dimension)


def _dtnk_position(solution) -> Tuple[float, float]:
    """(g x1, g x2) with g = 1 + sum_{i>=3} (xi - pi/2)^2."""
    g = 1.0
    for x in solution[2:]:
        g += (x - math.pi / 2) ** 2
    return solution[0] * g, solution[1] * g


class _SampledDynamicFrontMixin:
    """Grid-sampled front of the current environment, cached by t."""

    _GRID = 500

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        key = round(self.t, 9)
        cache = getattr(self, "_front_cache", None)
        if cache is None:
            cache = {}
            self._front_cache = cache

        if key not in cache:
            lower, upper = self.bounds
            a = np.linspace(lower[0], upper[0], self._GRID)
            b = np.linspace(lower[1], upper[1], self._GRID)
            X1, X2 = np.meshgrid(a, b)
            X1, X2 = X1.ravel(), X2.ravel()
            F, G = self._batch_evaluate(X1, X2)
            feasible = np.all(G <= 1e-9, axis=1)
            objectives = F[feasible]
            order = np.lexsort((objectives[:, 1], objectives[:, 0]))
            objectives = objectives[order]
            best_f2 = np.inf
            keep = np.zeros(len(objectives), dtype=bool)
            for i, (_, f2) in enumerate(objectives):
                if f2 < best_f2:
                    keep[i] = True
                    best_f2 = f2
            cache[key] = objectives[keep]

        front = cache[key]
        if len(front) > n_points:
            idx = np.linspace(0, len(front) - 1, n_points).astype(int)
            front = front[idx]
        return front.copy()


class DTNK(_SampledDynamicFrontMixin, _DynamicMOBase):
    """DTNK: static objectives with a dynamic constraint (SODC).

    The static TNK problem with an oscillating ring radius:

        min f1(x) = x1,  min f2(x) = x2
        s.t. x1^2 + x2^2 - r(t) - 0.1 cos(16 arctan(x1/x2)) >= 0,
             r(t) = 1 + 0.2 sin(0.5 pi t)
             (x1 - 0.5)^2 + (x2 - 0.5)^2 <= 0.5

    with x1, x2 in [0, pi]. The Pareto front lies on the ring constraint,
    which moves with r(t).
    """

    def __init__(self, tau_t: int = 10, n_t: int = 10, dimension: int = 2):
        super().__init__(
            dimension, _dtnk_bounds(dimension), "DTNK", tau_t, n_t
        )

    def _r(self) -> float:
        return 1.0 + 0.2 * math.sin(0.5 * math.pi * self.t)

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = _dtnk_position(solution)
        angle = math.atan2(x1, x2)
        g1 = -(x1**2 + x2**2 - self._r() - 0.1 * math.cos(16.0 * angle))
        g2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 - 0.5
        return Evaluation(
            fitness=[x1, x2], constraints_inequality=[g1, g2]
        )

    def _batch_evaluate(self, X1, X2):
        F = np.column_stack([X1, X2])
        angle = np.arctan2(X1, X2)
        G = np.column_stack([
            -(X1**2 + X2**2 - self._r() - 0.1 * np.cos(16.0 * angle)),
            (X1 - 0.5) ** 2 + (X2 - 0.5) ** 2 - 0.5,
        ])
        return F, G

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, True)


def _rotate(X1, X2, alpha):
    """Rotates the objective mapping f = (x1, x2) by alpha."""
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    return X1 * cos_a + X2 * sin_a, -X1 * sin_a + X2 * cos_a


class DTNK5(_SampledDynamicFrontMixin, _DynamicMOBase):
    """DTNK5: dynamic objectives with static constraints (DOSC).

    The static TNK constraints with a rotating objective mapping:

        min f1(x, t) =  x1 cos(alpha(t)) + x2 sin(alpha(t))
        min f2(x, t) = -x1 sin(alpha(t)) + x2 cos(alpha(t))
        alpha(t) = (pi / 8) sin(0.5 pi t)
        s.t. TNK's constraints (fixed ring, r = 1).

    The rotation changes which part of the fixed ring boundary is
    Pareto-optimal.
    """

    def __init__(self, tau_t: int = 10, n_t: int = 10, dimension: int = 2):
        super().__init__(
            dimension, _dtnk_bounds(dimension), "DTNK5", tau_t, n_t
        )

    def _alpha(self) -> float:
        return (math.pi / 8) * math.sin(0.5 * math.pi * self.t)

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = _dtnk_position(solution)
        f1, f2 = _rotate(x1, x2, self._alpha())
        angle = math.atan2(x1, x2)
        g1 = -(x1**2 + x2**2 - 1.0 - 0.1 * math.cos(16.0 * angle))
        g2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 - 0.5
        return Evaluation(fitness=[f1, f2], constraints_inequality=[g1, g2])

    def _batch_evaluate(self, X1, X2):
        F = np.column_stack(_rotate(X1, X2, self._alpha()))
        angle = np.arctan2(X1, X2)
        G = np.column_stack([
            -(X1**2 + X2**2 - 1.0 - 0.1 * np.cos(16.0 * angle)),
            (X1 - 0.5) ** 2 + (X2 - 0.5) ** 2 - 0.5,
        ])
        return F, G

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, False)


class DTNK6(_SampledDynamicFrontMixin, _DynamicMOBase):
    """DTNK6: dynamic objectives and dynamic constraints (DODC).

    DTNK's oscillating ring with DTNK5's rotating objective mapping:

        min f1(x, t) =  x1 cos(alpha(t)) + x2 sin(alpha(t))
        min f2(x, t) = -x1 sin(alpha(t)) + x2 cos(alpha(t))
        alpha(t) = (pi / 8) cos(0.5 pi t)
        s.t. x1^2 + x2^2 - r(t) - 0.1 cos(16 arctan(x1/x2)) >= 0,
             r(t) = 1 + 0.2 sin(0.5 pi t)
             (x1 - 0.5)^2 + (x2 - 0.5)^2 <= 0.5

    The ring radius moves the feasible boundary and the rotation changes
    which part of it is Pareto-optimal. The rotation uses cos so that the
    two changes are out of phase.
    """

    def __init__(self, tau_t: int = 10, n_t: int = 10, dimension: int = 2):
        super().__init__(
            dimension, _dtnk_bounds(dimension), "DTNK6", tau_t, n_t
        )

    def _r(self) -> float:
        return 1.0 + 0.2 * math.sin(0.5 * math.pi * self.t)

    def _alpha(self) -> float:
        return (math.pi / 8) * math.cos(0.5 * math.pi * self.t)

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = _dtnk_position(solution)
        f1, f2 = _rotate(x1, x2, self._alpha())
        angle = math.atan2(x1, x2)
        g1 = -(x1**2 + x2**2 - self._r() - 0.1 * math.cos(16.0 * angle))
        g2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 - 0.5
        return Evaluation(fitness=[f1, f2], constraints_inequality=[g1, g2])

    def _batch_evaluate(self, X1, X2):
        F = np.column_stack(_rotate(X1, X2, self._alpha()))
        angle = np.arctan2(X1, X2)
        G = np.column_stack([
            -(X1**2 + X2**2 - self._r() - 0.1 * np.cos(16.0 * angle)),
            (X1 - 0.5) ** 2 + (X2 - 0.5) ** 2 - 0.5,
        ])
        return F, G

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, True)


class DTNK4(_DynamicMOBase):
    """DTNK4: orbiting crescent with rotating objectives (DODC).

    A scaled TNK crescent (s = 0.5) whose centre (cx, cy) orbits (pi/2, pi/2)
    at radius 0.65, with its scallops rotating at twice the orbital rate, and
    objectives rotated by alpha(k):

        ring: (x1-cx)^2 + (x2-cy)^2 >= s^2 + 0.1 s^2 cos(16(theta - phi))
        disc: (x1-cx-0.5s)^2 + (x2-cy-0.5s)^2 <= 0.5 s^2
        f1 = x1 cos(alpha) + x2 sin(alpha)
        f2 = -x1 sin(alpha) + x2 cos(alpha),  alpha(k) = (pi/8) sin(2 pi k / K)

    The shape depends on the environment index k = floor(tau / tau_t) only,
    with period K = 20 environments, so n_t has no effect.
    """

    _GRID = 1000
    _SCALE = 0.5
    _ORBIT_R = 0.65
    _ORBIT_CENTRE = (math.pi / 2, math.pi / 2)
    _K = 20

    def __init__(self, tau_t: int = 10, n_t: int = 10, dimension: int = 2):
        super().__init__(
            dimension, _dtnk_bounds(dimension), "DTNK4", tau_t, n_t
        )
        self._front_cache = {}

    @property
    def _k(self) -> int:
        """Environment index (integer count of environment changes)."""
        return self._tau // self.tau_t

    def _phase(self) -> float:
        """Orbital phase driven by environment index."""
        return 2.0 * math.pi * self._k / self._K

    def _centre(self) -> Tuple[float, float]:
        """Ring centre position on the orbit."""
        phase = self._phase()
        return (
            self._ORBIT_CENTRE[0] + self._ORBIT_R * math.cos(phase),
            self._ORBIT_CENTRE[1] + self._ORBIT_R * math.sin(phase),
        )

    def _scallop_phase(self) -> float:
        """Scallop rotation: twice the orbital rate."""
        return self._phase() * 2.0

    def _alpha(self) -> float:
        """Objective rotation angle, oscillates around 0."""
        return (math.pi / 8) * math.sin(self._phase())

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = _dtnk_position(solution)
        s = self._SCALE
        cx, cy = self._centre()
        phi = self._scallop_phase()
        alpha = self._alpha()

        dx = cx + 0.5 * s
        dy = cy + 0.5 * s

        rx, ry = x1 - cx, x2 - cy
        angle = math.atan2(rx, ry)
        dist_sq = rx ** 2 + ry ** 2
        ring_r = s ** 2 + 0.1 * s ** 2 * math.cos(16.0 * (angle - phi))
        g1 = -(dist_sq - ring_r)
        g2 = (x1 - dx) ** 2 + (x2 - dy) ** 2 - 0.5 * s ** 2

        f1 = x1 * math.cos(alpha) + x2 * math.sin(alpha)
        f2 = -x1 * math.sin(alpha) + x2 * math.cos(alpha)

        return Evaluation(
            fitness=[f1, f2], constraints_inequality=[g1, g2]
        )

    def _batch_evaluate(self, X1, X2):
        s = self._SCALE
        cx, cy = self._centre()
        phi = self._scallop_phase()
        alpha = self._alpha()

        dx = cx + 0.5 * s
        dy = cy + 0.5 * s

        RX, RY = X1 - cx, X2 - cy
        angle = np.arctan2(RX, RY)
        dist_sq = RX ** 2 + RY ** 2
        ring_r = s ** 2 + 0.1 * s ** 2 * np.cos(16.0 * (angle - phi))

        G = np.column_stack([
            -(dist_sq - ring_r),
            (X1 - dx) ** 2 + (X2 - dy) ** 2 - 0.5 * s ** 2,
        ])

        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        F = np.column_stack([
            X1 * cos_a + X2 * sin_a,
            -X1 * sin_a + X2 * cos_a,
        ])
        return F, G

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        """Grid-sampled reference front, cached by environment index."""
        key = self._k
        if key not in self._front_cache:
            lower, upper = self.bounds
            a = np.linspace(lower[0], upper[0], self._GRID)
            b = np.linspace(lower[1], upper[1], self._GRID)
            X1, X2 = np.meshgrid(a, b)
            X1, X2 = X1.ravel(), X2.ravel()
            F, G = self._batch_evaluate(X1, X2)
            feasible = np.all(G <= 1e-9, axis=1)
            objectives = F[feasible]
            order = np.lexsort((objectives[:, 1], objectives[:, 0]))
            objectives = objectives[order]
            best_f2 = np.inf
            keep = np.zeros(len(objectives), dtype=bool)
            for i, (_, f2) in enumerate(objectives):
                if f2 < best_f2:
                    keep[i] = True
                    best_f2 = f2
            self._front_cache[key] = objectives[keep]

        front = self._front_cache[key]
        if len(front) > n_points:
            idx = np.linspace(0, len(front) - 1, n_points).astype(int)
            front = front[idx]
        return front.copy()

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, True)
