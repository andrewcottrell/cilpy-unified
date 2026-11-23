# cilpy/compare/metrics.py
"""Performance metrics for multi-objective optimization (minimization).

GD (Van Veldhuizen and Lamont, 1998), IGD (Coello Coello and Reyes Sierra,
2004), hypervolume (Zitzler and Thiele, 1999), spacing (Schott, 1995) and
spread (Deb et al., 2002), plus the hypervolume ratio and P_RED used as the
relative error, and the feasibility rate.
"""

from typing import Optional, Sequence

import numpy as np

from cilpy.problem import Evaluation


def _as_matrix(front: Sequence[Sequence[float]]) -> np.ndarray:
    """Coerces a front to a 2-D float array of shape (n_solutions, n_objectives)."""
    matrix = np.asarray(front, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def nondominated_filter(front: Sequence[Sequence[float]]) -> np.ndarray:
    """Reduces a set of objective vectors to its non-dominated subset.

    Args:
        front: An (n, nm) collection of objective vectors (minimization).

    Returns:
        The non-dominated subset as an (k, nm) array, k <= n.
    """
    matrix = _as_matrix(front)
    n = matrix.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        others = np.delete(matrix, i, axis=0)
        dominated = np.any(
            np.all(others <= matrix[i], axis=1)
            & np.any(others < matrix[i], axis=1)
        )
        if dominated:
            keep[i] = False
    return matrix[keep]


def generational_distance(
    front: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
    p: float = 2.0,
) -> float:
    """Generational Distance (GD): convergence to the reference front.

    GD = ( (1/n) * sum_i d_i^p )^(1/p), where d_i is the Euclidean distance
    from obtained solution i to its nearest reference-front point.

    A GD of 0 means every obtained solution lies on the reference front.

    Args:
        front: The obtained front, shape (n, nm).
        reference_front: A dense sampling of the true Pareto front.
        p: The norm exponent. Defaults to 2 (the common convention).

    Returns:
        The generational distance (lower is better, 0 is ideal).
    """
    obtained = _as_matrix(front)
    reference = _as_matrix(reference_front)
    dists = np.min(
        np.linalg.norm(obtained[:, None, :] - reference[None, :, :], axis=2),
        axis=1,
    )
    return float((np.mean(dists**p)) ** (1.0 / p))


def inverted_generational_distance(
    front: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
    p: float = 2.0,
) -> float:
    """Inverted Generational Distance (IGD): convergence and coverage.

    IGD = ( (1/m) * sum_j d_j^p )^(1/p), where d_j is the Euclidean distance
    from reference-front point j to its nearest obtained solution.

    Args:
        front: The obtained front, shape (n, nm).
        reference_front: A dense sampling of the true Pareto front.
        p: The norm exponent. Defaults to 2.

    Returns:
        The inverted generational distance (lower is better, 0 is ideal).
    """
    obtained = _as_matrix(front)
    reference = _as_matrix(reference_front)
    dists = np.min(
        np.linalg.norm(reference[:, None, :] - obtained[None, :, :], axis=2),
        axis=1,
    )
    return float((np.mean(dists**p)) ** (1.0 / p))


def hypervolume(
    front: Sequence[Sequence[float]],
    reference_point: Sequence[float],
) -> float:
    """Hypervolume (HV) of a bi-objective front (exact, O(n log n)).

    The area of objective space dominated by the front and bounded above by
    `reference_point`.

    Args:
        front: The obtained front, shape (n, 2). Solutions that do not
            dominate the reference point contribute nothing.
        reference_point: A point [r1, r2] dominated by the region of
            interest (e.g. slightly worse than the nadir point).

    Returns:
        The dominated area (higher is better).

    Raises:
        NotImplementedError: If the front has more than two objectives.
    """
    matrix = _as_matrix(front)
    if matrix.shape[1] != 2:
        raise NotImplementedError(
            "hypervolume() supports bi-objective fronts only."
        )
    r1, r2 = float(reference_point[0]), float(reference_point[1])

    # Keep only points that strictly dominate the reference point.
    mask = (matrix[:, 0] < r1) & (matrix[:, 1] < r2)
    points = nondominated_filter(matrix[mask])
    if points.size == 0:
        return 0.0

    # Sort by f1 ascending; f2 is then strictly descending on a ND set.
    points = points[np.argsort(points[:, 0])]

    area = 0.0
    prev_f2 = r2
    for f1, f2 in points:
        area += (r1 - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return float(area)


def reference_point_from_front(
    reference_front: Sequence[Sequence[float]],
    margin: float = 0.1,
) -> np.ndarray:
    """Reference point at the worst value of each objective on the reference
    front, plus margin times the front's width.

    Args:
        reference_front: A sampling of the true Pareto front.
        margin: Fraction of the front's width added. Defaults to 0.1.

    Returns:
        The reference point.
    """
    reference = _as_matrix(reference_front)
    ideal = reference.min(axis=0)
    nadir = reference.max(axis=0)
    width = np.where(nadir > ideal, nadir - ideal, 1.0)
    return nadir + margin * width


def hypervolume_ratio(
    front: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
    reference_point: Optional[Sequence[float]] = None,
) -> float:
    """HV(front) / HV(reference_front), both from the same reference point,
    clipped to [0, 1].

    Args:
        front: The obtained front.
        reference_front: A sampling of the true Pareto front.
        reference_point: Defaults to reference_point_from_front(reference_front).

    Returns:
        The ratio; 0 for an empty front.

    Raises:
        ValueError: If the reference front has zero hypervolume.
    """
    if len(front) == 0:
        return 0.0
    if reference_point is None:
        reference_point = reference_point_from_front(reference_front)
    best_possible = hypervolume(reference_front, reference_point)
    if best_possible <= 0:
        raise ValueError("Reference front has zero hypervolume.")
    achieved = hypervolume(front, reference_point)
    return float(min(max(achieved / best_possible, 0.0), 1.0))


def p_red(scores: Sequence[float]) -> float:
    """P_RED = sqrt(mean((1 - b)^2)) over per-iteration scores b, where
    b = 1 is optimal.

    Args:
        scores: The per-iteration scores b.

    Returns:
        P_RED; 0 when every score is 1.
    """
    b = np.asarray(scores, dtype=float)
    return float(np.sqrt(np.mean((1.0 - b) ** 2)))


def spacing(front: Sequence[Sequence[float]]) -> float:
    """Schott's spacing metric: uniformity of the front's distribution.

    S = sqrt( (1/(n-1)) * sum_i (d_bar - d_i)^2 ), where d_i is the
    Manhattan distance from solution i to its nearest neighbour in the
    front and d_bar is the mean of the d_i.

    A spacing of 0 means all solutions are equidistant from their nearest
    neighbours.

    Args:
        front: The obtained front, shape (n, nm), n >= 2.

    Returns:
        The spacing value (lower is better, 0 is ideal).
    """
    matrix = _as_matrix(front)
    n = matrix.shape[0]
    if n < 2:
        return 0.0

    dist = np.sum(
        np.abs(matrix[:, None, :] - matrix[None, :, :]), axis=2
    )
    np.fill_diagonal(dist, np.inf)
    d = np.min(dist, axis=1)
    d_bar = np.mean(d)
    return float(np.sqrt(np.sum((d_bar - d) ** 2) / (n - 1)))


def spread(
    front: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
) -> float:
    """Deb's Delta spread metric for bi-objective fronts.

    Delta = (d_f + d_l + sum_i |d_i - d_bar|) / (d_f + d_l + (n-1) d_bar)

    where d_f and d_l are the distances between the extreme points of the
    reference front and the boundary solutions of the obtained front, d_i
    are consecutive-neighbour distances along the obtained front (sorted by
    the first objective), and d_bar is their mean.

    Args:
        front: The obtained front, shape (n, 2), n >= 2.
        reference_front: A sampling of the true Pareto front, used only for
            its extreme points.

    Returns:
        The spread value (lower is better, 0 is ideal).

    Raises:
        NotImplementedError: If the front has more than two objectives.
    """
    matrix = _as_matrix(front)
    if matrix.shape[1] != 2:
        raise NotImplementedError(
            "spread() supports bi-objective fronts only."
        )
    reference = _as_matrix(reference_front)

    matrix = matrix[np.argsort(matrix[:, 0])]
    reference = reference[np.argsort(reference[:, 0])]

    n = matrix.shape[0]
    if n < 2:
        return 1.0

    d_f = float(np.linalg.norm(matrix[0] - reference[0]))
    d_l = float(np.linalg.norm(matrix[-1] - reference[-1]))

    consecutive = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
    d_bar = float(np.mean(consecutive))

    numerator = d_f + d_l + float(np.sum(np.abs(consecutive - d_bar)))
    denominator = d_f + d_l + (n - 1) * d_bar
    if denominator == 0:
        return 0.0
    return numerator / denominator


def feasibility_rate(
    evaluations: Sequence[Evaluation],
    tolerance: float = 1e-6,
) -> float:
    """Percentage of evaluations satisfying all constraints.

    A solution is feasible when every inequality constraint value is <= 0
    and every equality constraint value is within `tolerance` of 0.
    Unconstrained evaluations count as feasible.

    Args:
        evaluations: `Evaluation` objects (e.g. from
            `Solver.get_population_evaluations()` or an archive).
        tolerance: Absolute tolerance for equality constraints.

    Returns:
        Feasibility rate in [0, 100].
    """
    if not evaluations:
        return 0.0

    def _is_feasible(evaluation: Evaluation) -> bool:
        if evaluation.constraints_inequality:
            if any(v > 0 for v in evaluation.constraints_inequality):
                return False
        if evaluation.constraints_equality:
            if any(abs(v) > tolerance for v in evaluation.constraints_equality):
                return False
        return True

    feasible = sum(1 for e in evaluations if _is_feasible(e))
    return 100.0 * feasible / len(evaluations)
