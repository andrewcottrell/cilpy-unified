# cilpy/solver/mgpso.py
"""Multi-guide particle swarm optimisation (MGPSO).

One sub-swarm per objective, a shared bounded archive of non-dominated
solutions, and an archive guide in the velocity update (Scheepers,
Engelbrecht and Cleghorn, 2019). The control parameters are fixed when w,
c1, c2 and c3 are given, and otherwise sampled per particle every
iteration under the stability condition of the paper.
"""

import copy
from typing import List, Optional, Tuple

import numpy as np

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler


def dominates(fitness_a: List[float], fitness_b: List[float]) -> bool:
    """Pareto dominance for minimization.

    Solution `a` dominates solution `b` if `a` is no worse than `b` in all
    objectives and strictly better in at least one.

    Args:
        fitness_a: Objective vector of the first solution.
        fitness_b: Objective vector of the second solution.

    Returns:
        True if `fitness_a` dominates `fitness_b`.
    """
    no_worse = all(a <= b for a, b in zip(fitness_a, fitness_b))
    strictly_better = any(a < b for a, b in zip(fitness_a, fitness_b))
    return no_worse and strictly_better


def _is_feasible(evaluation: Evaluation, tolerance: float = 1e-6) -> bool:
    """True if all inequality constraints are <= 0 and all equality
    constraints are within tolerance of 0. Unconstrained evaluations are
    feasible by definition."""
    if evaluation.constraints_inequality:
        if any(g > 0 for g in evaluation.constraints_inequality):
            return False
    if evaluation.constraints_equality:
        if any(abs(h) > tolerance for h in evaluation.constraints_equality):
            return False
    return True


class _Archive:
    """Bounded archive of mutually non-dominated (position, Evaluation) pairs.

    When an insertion exceeds capacity, the member with the smallest crowding
    distance is removed. The objective matrix and crowding distances are
    cached until the archive changes.
    """

    def __init__(self, capacity: int, feasible_only: bool = False):
        self.capacity = capacity
        self.feasible_only = feasible_only
        self.positions: List[np.ndarray] = []
        self.evaluations: List[Evaluation] = []
        self._matrix_cache: Optional[np.ndarray] = None
        self._crowding_cache: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def entries(self) -> List[Tuple[np.ndarray, Evaluation]]:
        return list(zip(self.positions, self.evaluations))

    def _invalidate(self) -> None:
        self._matrix_cache = None
        self._crowding_cache = None

    def _matrix(self) -> np.ndarray:
        """The (n, nm) objective matrix of the archive, cached."""
        if self._matrix_cache is None:
            self._matrix_cache = np.array(
                [e.fitness for e in self.evaluations], dtype=float
            )
        return self._matrix_cache

    def insert(self, position: np.ndarray, evaluation: Evaluation) -> None:
        """Attempts to insert a solution into the archive.

        The solution is rejected if it is dominated by any archive member.
        Otherwise, all archive members it dominates are removed and the
        solution is added. If the archive then exceeds capacity, the most
        crowded member is evicted.
        """
        # feasible_only: reject infeasible solutions.
        if self.feasible_only and not _is_feasible(evaluation):
            return

        fitness = np.asarray(evaluation.fitness, dtype=float)

        if self.positions:
            matrix = self._matrix()

            # Reject if dominated by any existing member.
            dominated_by_existing = np.any(
                np.all(matrix <= fitness, axis=1)
                & np.any(matrix < fitness, axis=1)
            )
            if dominated_by_existing:
                return

            # Remove members dominated by the new solution.
            dominates_mask = np.all(fitness <= matrix, axis=1) & np.any(
                fitness < matrix, axis=1
            )
            if dominates_mask.any():
                keep = ~dominates_mask
                self.positions = [
                    p for p, k in zip(self.positions, keep) if k
                ]
                self.evaluations = [
                    e for e, k in zip(self.evaluations, keep) if k
                ]

        self.positions.append(position.copy())
        self.evaluations.append(copy.deepcopy(evaluation))
        self._invalidate()

        if len(self) > self.capacity:
            self._evict_most_crowded()

    def rescore(self, evaluations: List[Evaluation]) -> None:
        """Replaces the stored evaluations; positions are kept.

        Args:
            evaluations: New evaluations, in the order of `positions`.
        """
        self.evaluations = [copy.deepcopy(e) for e in evaluations]
        self._invalidate()

    def prune(self) -> Tuple[int, int]:
        """Removes infeasible members (when feasible_only) and dominated members.

        Returns:
            (n_infeasible_removed, n_dominated_removed).
        """
        n = len(self)
        if n == 0:
            return (0, 0)

        keep = np.ones(n, dtype=bool)

        n_infeasible = 0
        if self.feasible_only:
            for i, evaluation in enumerate(self.evaluations):
                if not _is_feasible(evaluation):
                    keep[i] = False
                    n_infeasible += 1

        n_dominated = 0
        surviving = np.flatnonzero(keep)
        if surviving.size > 1:
            matrix = self._matrix()[surviving]
            # dominated[i] is True if any other surviving member dominates i
            no_worse = np.all(matrix[None, :, :] <= matrix[:, None, :], axis=2)
            better = np.any(matrix[None, :, :] < matrix[:, None, :], axis=2)
            dominated = np.any(no_worse & better, axis=1)
            n_dominated = int(dominated.sum())
            keep[surviving[dominated]] = False

        if not keep.all():
            self.positions = [p for p, k in zip(self.positions, keep) if k]
            self.evaluations = [e for e, k in zip(self.evaluations, keep) if k]
            self._invalidate()

        return (n_infeasible, n_dominated)

    def _crowding_distances(self) -> np.ndarray:
        """Crowding distance of every member; boundary members get infinity.

        Cached until the archive changes.
        """
        if self._crowding_cache is not None:
            return self._crowding_cache

        n = len(self)
        if n == 0:
            distances = np.array([])
        elif n <= 2:
            distances = np.full(n, np.inf)
        else:
            matrix = self._matrix()
            nm = matrix.shape[1]
            distances = np.zeros(n)
            for m in range(nm):
                sorted_idx = np.argsort(matrix[:, m])
                sorted_col = matrix[sorted_idx, m]
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_range = sorted_col[-1] - sorted_col[0]
                if f_range == 0:
                    continue
                distances[sorted_idx[1:-1]] += (
                    sorted_col[2:] - sorted_col[:-2]
                ) / f_range

        self._crowding_cache = distances
        return distances

    def _evict_most_crowded(self) -> None:
        """Removes the member with the smallest crowding distance."""
        distances = self._crowding_distances()
        most_crowded_idx = int(np.argmin(distances))
        self.positions.pop(most_crowded_idx)
        self.evaluations.pop(most_crowded_idx)
        self._invalidate()

    def tournament_select(self, tournament_size: int) -> Optional[np.ndarray]:
        """Selects an archive guide via crowding-distance tournament.

        A tournament of `tournament_size` random members is held and the
        member with the largest crowding distance wins.

        Returns:
            The winning position, or None if the archive is empty.
        """
        n = len(self)
        if n == 0:
            return None

        k = min(tournament_size, n)
        indices = np.random.choice(n, size=k, replace=False)
        distances = self._crowding_distances()
        winner = indices[int(np.argmax(distances[indices]))]
        return self.positions[winner].copy()


class _Particle:
    """A single MGPSO particle.

    Holds position, velocity, personal best (position and its `Evaluation` on
    the sub-swarm's assigned objective), the fixed per-particle lambda, and
    the current control parameters used by stability-guided sampling.
    """

    MAX_SAMPLE_ATTEMPTS = 10

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.lower = lower
        self.upper = upper
        self.nx = lower.shape[0]

        self.x = np.random.uniform(lower, upper)
        self.v = np.zeros(self.nx)

        # Personal best on the assigned objective (scalar Evaluation).
        self.y = self.x.copy()
        self.y_eval: Optional[Evaluation] = None

        # Lambda: sampled once, fixed for the particle's lifetime.
        self.lam = np.random.uniform(0.0, 1.0)

        # Current control parameters (reused if stability sampling fails).
        self.w = 0.5
        self.c1 = 1.0
        self.c2 = 1.0
        self.c3 = 1.0

    def _satisfies_stability(
        self, w: float, c1: float, c2: float, c3: float
    ) -> bool:
        """Checks the MGPSO order-1/order-2 stability condition.

        (Scheepers et al., 2019, Eq. 18):

            0 < rho < 4(1 - w^2) /
                (1 - w + (c1^2 + lam^2 c2^2 + (1-lam)^2 c3^2)(1+w) / (3 rho^2))

        where rho = c1 + lam*c2 + (1 - lam)*c3 and |w| < 1.
        """
        if abs(w) >= 1.0:
            return False

        lam = self.lam
        rho = c1 + lam * c2 + (1.0 - lam) * c3
        if rho <= 0:
            return False

        numerator = 4.0 * (1.0 - w**2)
        denom_term = (
            c1**2 + lam**2 * c2**2 + (1.0 - lam) ** 2 * c3**2
        ) * (1.0 + w)
        denominator = 1.0 - w + denom_term / (3.0 * rho**2)

        return rho < numerator / denominator

    def sample_stable_params(self) -> None:
        """Samples w ~ U(0,1) and c1, c2, c3 ~ U(0,2) until stable.

        Limited to `MAX_SAMPLE_ATTEMPTS`; the previous parameters are kept
        if every attempt fails.
        """
        for _ in range(self.MAX_SAMPLE_ATTEMPTS):
            w = np.random.uniform(0.0, 1.0)
            c1 = np.random.uniform(0.0, 2.0)
            c2 = np.random.uniform(0.0, 2.0)
            c3 = np.random.uniform(0.0, 2.0)
            if self._satisfies_stability(w, c1, c2, c3):
                self.w, self.c1, self.c2, self.c3 = w, c1, c2, c3
                return

    def set_fixed_params(self, w: float, c1: float, c2: float, c3: float) -> None:
        self.w, self.c1, self.c2, self.c3 = w, c1, c2, c3

    def update(self, y_hat: np.ndarray, a_hat: Optional[np.ndarray]) -> None:
        """One velocity and position update (Scheepers et al., 2019, Eq. 3).

        Args:
            y_hat: The sub-swarm neighbourhood best position.
            a_hat: The archive guide position, or None if the archive is
                empty, in which case the personal best is used as fallback.
        """
        if a_hat is None:
            a_hat = self.y

        r1 = np.random.uniform(0.0, 1.0, self.nx)
        r2 = np.random.uniform(0.0, 1.0, self.nx)
        r3 = np.random.uniform(0.0, 1.0, self.nx)
        lam = self.lam

        self.v = (
            self.w * self.v
            + self.c1 * r1 * (self.y - self.x)
            + lam * self.c2 * r2 * (y_hat - self.x)
            + (1.0 - lam) * self.c3 * r3 * (a_hat - self.x)
        )
        self.x = np.clip(self.x + self.v, self.lower, self.upper)


class _Subswarm:
    """A sub-swarm responsible for a single objective.

    Uses a global (star) neighbourhood topology: every particle shares the
    same neighbourhood best.
    """

    def __init__(
        self,
        n_particles: int,
        lower: np.ndarray,
        upper: np.ndarray,
        objective_idx: int,
    ):
        self.objective_idx = objective_idx
        self.particles = [_Particle(lower, upper) for _ in range(n_particles)]
        self.evaluations: List[Evaluation] = []

        self.y_hat: np.ndarray = self.particles[0].x.copy()
        self.y_hat_eval: Optional[Evaluation] = None

    def _scalar_eval(self, evaluation: Evaluation) -> Evaluation:
        """This sub-swarm's objective value as an Evaluation; constraint
        values are kept."""
        return Evaluation(
            fitness=float(evaluation.fitness[self.objective_idx]),
            constraints_inequality=evaluation.constraints_inequality,
            constraints_equality=evaluation.constraints_equality,
        )

    def evaluate(self, problem: Problem) -> None:
        """Evaluates every particle against the full problem."""
        self.evaluations = [
            problem.evaluate(list(p.x)) for p in self.particles
        ]

    def update_bests(self, comparator: ConstraintHandler) -> None:
        """Updates the personal and neighbourhood bests on the assigned
        objective with the comparator."""
        for particle, evaluation in zip(self.particles, self.evaluations):
            scalar = self._scalar_eval(evaluation)

            if particle.y_eval is None or comparator.is_better(
                scalar, particle.y_eval
            ):
                particle.y = particle.x.copy()
                particle.y_eval = scalar

            if self.y_hat_eval is None or comparator.is_better(
                scalar, self.y_hat_eval
            ):
                self.y_hat = particle.x.copy()
                self.y_hat_eval = scalar

    def update_archive(self, archive: _Archive) -> None:
        """Offers every particle's current solution to the shared archive."""
        for particle, evaluation in zip(self.particles, self.evaluations):
            archive.insert(particle.x, evaluation)

    def update_particles(
        self,
        archive: _Archive,
        tournament_size: int,
        fixed_params: Optional[Tuple[float, float, float, float]],
    ) -> None:
        """Updates velocity and position of every particle."""
        for particle in self.particles:
            if fixed_params is None:
                particle.sample_stable_params()
            else:
                particle.set_fixed_params(*fixed_params)
            a_hat = archive.tournament_select(tournament_size)
            particle.update(self.y_hat, a_hat)


class MGPSO(Solver[List[float], List[float]]):
    """Multi-Guide Particle Swarm Optimization solver.

    One sub-swarm is created per objective; all sub-swarms share a bounded
    non-dominated archive whose members act as additional velocity guides.

    Attributes:
        n_objectives (int): The number of objectives, determined by probing
            the problem once at initialisation.
        swarm_size (int): Number of particles *per sub-swarm*. The total
            population is `n_objectives * swarm_size`, which is also the
            archive capacity.

    Example:
            from cilpy.problem.multi_objective import ZDT1
            from cilpy.solver.mgpso import MGPSO
            from cilpy.runner import ExperimentRunner

            runner = ExperimentRunner(
                problems=[ZDT1()],
                solver_configurations=[{
                    "class": MGPSO,
                    "params": {"name": "MGPSO", "swarm_size": 50},
                }],
                num_runs=30,
                max_iterations=1000,
            )
            runner.run_experiments()
    """

    def __init__(
        self,
        problem: Problem[List[float], List[float]],
        name: str,
        swarm_size: int,
        tournament_size: int = 3,
        feasible_archive_only: bool = False,
        refresh_mode: str = "clean",
        sentry_mode: str = "fixed",
        n_sentries: Optional[int] = None,
        w: Optional[float] = None,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        c3: Optional[float] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        """Initializes the MGPSO solver.

        Args:
            problem: The multi-objective optimization problem to solve. Its
                `evaluate` method must return an `Evaluation` whose fitness is
                a list of floats.
            name: The name of the solver instance.
            swarm_size: The number of particles in each sub-swarm.
            tournament_size: Archive-guide tournament size (2 or 3 in the
                original paper). Defaults to 3.
            feasible_archive_only: If True, infeasible solutions are not
                admitted to the archive.
            refresh_mode: Archive update on an environment change. "clean"
                (default) re-scores the members and keeps those that stay
                feasible and non-dominated; "clear" empties the archive.
            sentry_mode: Change detection. "fixed" (default) re-evaluates the
                lower corner of the search space; "archive" re-evaluates the
                archive members, or the fixed sentry when the archive is
                empty.
            n_sentries: Archive members checked per step in archive mode.
                None (default) checks all of them.
            w: Inertia weight. If `w`, `c1`, `c2`, and `c3` are all provided,
                fixed control parameters are used; otherwise parameters are
                re-sampled per particle each iteration subject to the MGPSO
                stability condition.
            c1: Cognitive (personal best) acceleration coefficient.
            c2: Social (neighbourhood best) acceleration coefficient.
            c3: Archive-guide acceleration coefficient.
            constraint_handler: Optional comparator used for personal and
                neighbourhood best updates. Defaults to fitness-only
                comparison.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(problem, name, constraint_handler=constraint_handler)

        self.swarm_size = swarm_size
        self.tournament_size = tournament_size

        if refresh_mode not in ("clean", "clear"):
            raise ValueError(
                f"refresh_mode must be 'clean' or 'clear', got "
                f"'{refresh_mode}'."
            )
        self.refresh_mode = refresh_mode

        if sentry_mode not in ("fixed", "archive"):
            raise ValueError(
                f"sentry_mode must be 'fixed' or 'archive', got "
                f"'{sentry_mode}'."
            )
        self.sentry_mode = sentry_mode
        self.n_sentries = n_sentries
        self.n_changes_detected = 0

        fixed = (w, c1, c2, c3)
        if all(p is not None for p in fixed):
            self._fixed_params: Optional[Tuple[float, float, float, float]] = (
                float(w), float(c1), float(c2), float(c3)
            )
        else:
            self._fixed_params = None

        lower_list, upper_list = self.problem.bounds
        lower = np.asarray(lower_list, dtype=float)
        upper = np.asarray(upper_list, dtype=float)

        # Cilpy problems do not expose an objective count; probe once.
        probe = self.problem.evaluate(list(lower))
        if not isinstance(probe.fitness, (list, tuple, np.ndarray)):
            raise ValueError(
                f"MGPSO requires a multi-objective problem, but "
                f"'{self.problem.name}' returned a scalar fitness."
            )
        self.n_objectives = len(probe.fitness)

        self._archive = _Archive(
            capacity=self.n_objectives * self.swarm_size,
            feasible_only=feasible_archive_only,
        )
        self._swarms = [
            _Subswarm(self.swarm_size, lower, upper, objective_idx=m)
            for m in range(self.n_objectives)
        ]

        # Fixed sentry: the lower corner, re-evaluated each step on dynamic
        # problems.
        self._sentry_x = lower.copy()
        self._sentry_eval = probe

        # Evaluate the initial population so that get_result() and the
        # population accessors are valid before the first step() call.
        self._dynamic = any(self.problem.is_dynamic())
        self._environment_checked = False
        for swarm in self._swarms:
            swarm.evaluate(self.problem)
            swarm.update_bests(self.comparator)
            swarm.update_archive(self._archive)

    def _refresh_archive(
        self, refresh_mode: Optional[str] = None, evaluate=None
    ) -> None:
        """Re-evaluates the archive after a change.

        "clean" re-scores every member and prunes infeasible (when
        feasible_only) and dominated members; "clear" empties the archive.
        refresh_mode and evaluate override self.refresh_mode and
        self.problem.evaluate for this call.
        """
        if len(self._archive) == 0:
            return

        if (refresh_mode or self.refresh_mode) == "clear":
            self._archive.positions = []
            self._archive.evaluations = []
            self._archive._invalidate()
            return

        evaluate = evaluate or self.problem.evaluate
        evaluations = [
            evaluate(list(position))
            for position in self._archive.positions
        ]
        self._archive.rescore(evaluations)
        self._archive.prune()

    @staticmethod
    def _evaluations_differ(
        fresh: Evaluation, old: Evaluation, tolerance: float
    ) -> bool:
        """True if the fitness or any constraint value moved by more than
        tolerance."""
        if np.any(np.abs(
            np.asarray(fresh.fitness) - np.asarray(old.fitness)
        ) > tolerance):
            return True
        for new_c, old_c in (
            (fresh.constraints_inequality, old.constraints_inequality),
            (fresh.constraints_equality, old.constraints_equality),
        ):
            if new_c and old_c and np.any(
                np.abs(np.asarray(new_c) - np.asarray(old_c)) > tolerance
            ):
                return True
        return False

    def _fixed_sentry_changed(self, tolerance: float) -> bool:
        """Re-evaluates the single fixed sentry (the lower corner)."""
        fresh = self.problem.evaluate(list(self._sentry_x))
        old = self._sentry_eval
        self._sentry_eval = fresh
        return self._evaluations_differ(fresh, old, tolerance)

    def _archive_sentries_changed(self, tolerance: float) -> bool:
        """Re-evaluates archive members and compares them with their stored
        evaluations. Falls back to the fixed sentry if the archive is empty.
        On a detected change the fixed sentry is re-evaluated as well."""
        n = len(self._archive)
        if n == 0:
            return self._fixed_sentry_changed(tolerance)

        if self.n_sentries is None or self.n_sentries >= n:
            indices = range(n)
        else:
            indices = np.random.choice(n, size=self.n_sentries, replace=False)

        for i in indices:
            position = self._archive.positions[i]
            stored = self._archive.evaluations[i]
            fresh = self.problem.evaluate(list(position))
            if self._evaluations_differ(fresh, stored, tolerance):
                self._sentry_eval = self.problem.evaluate(list(self._sentry_x))
                return True
        return False

    def _environment_changed(self, tolerance: float = 1e-12) -> bool:
        """Detects an environment change using the chosen sentry mode."""
        if self.sentry_mode == "archive":
            changed = self._archive_sentries_changed(tolerance)
        else:
            changed = self._fixed_sentry_changed(tolerance)
        if changed:
            self.n_changes_detected += 1
        return changed

    def respond_to_change(self) -> None:
        """Re-evaluates the archive and personal bests. Called by a wrapping
        solver."""
        self._respond_to_change()

    def check_environment(self) -> None:
        """Detects an environment change and responds to it, at most once per
        iteration. Called by `step`, or earlier by a wrapping solver that must
        see the environment change before it changes the landscape itself."""
        if self._environment_checked:
            return
        self._environment_checked = True
        if self._dynamic and self._environment_changed():
            # A caching proxy problem (CCLS) drops its cache after a change.
            clear_cache = getattr(self.problem, "clear_cache", None)
            if clear_cache is not None:
                clear_cache()
            self._respond_to_change()

    def rescore(self) -> None:
        """Re-scores the archive, personal bests and sentry in place, without
        counting an environment change. Called by
        CoevolutionaryLagrangianSolver when its multipliers change; uses
        problem.evaluate_cached when the problem has it."""
        evaluate = getattr(self.problem, "evaluate_cached", self.problem.evaluate)
        self._respond_to_change(refresh_mode="clean", evaluate=evaluate)
        self._sentry_eval = evaluate(list(self._sentry_x))

    def _respond_to_change(
        self, refresh_mode: Optional[str] = None, evaluate=None
    ) -> None:
        """Refreshes the archive, re-evaluates every personal best, and
        recomputes each sub-swarm's neighbourhood best."""
        evaluate = evaluate or self.problem.evaluate
        self._refresh_archive(refresh_mode, evaluate)

        for swarm in self._swarms:
            swarm.y_hat_eval = None
            for particle in swarm.particles:
                if particle.y_eval is None:
                    continue
                full_eval = evaluate(list(particle.y))
                particle.y_eval = swarm._scalar_eval(full_eval)
                if swarm.y_hat_eval is None or self.comparator.is_better(
                    particle.y_eval, swarm.y_hat_eval
                ):
                    swarm.y_hat = particle.y.copy()
                    swarm.y_hat_eval = particle.y_eval

    def step(self) -> None:
        """Performs one iteration: checks for an environment change, moves
        every particle, then evaluates each sub-swarm, updates the bests and
        offers the new positions to the archive."""
        self.check_environment()

        for swarm in self._swarms:
            swarm.update_particles(
                self._archive, self.tournament_size, self._fixed_params
            )

        for swarm in self._swarms:
            swarm.evaluate(self.problem)
            swarm.update_bests(self.comparator)
            swarm.update_archive(self._archive)
        self._environment_checked = False

    def get_result(self) -> List[Tuple[List[float], Evaluation[List[float]]]]:
        """Returns the archive of non-dominated solutions (the Pareto front).

        Returns:
            A list of `(solution, Evaluation)` tuples, one per archive member.
        """
        return [
            (position.tolist(), copy.deepcopy(evaluation))
            for position, evaluation in self._archive.entries
        ]

    def get_population(self) -> List[List[float]]:
        """Returns the positions of all particles across all sub-swarms."""
        return [
            particle.x.tolist()
            for swarm in self._swarms
            for particle in swarm.particles
        ]

    def get_population_evaluations(self) -> List[Evaluation[List[float]]]:
        """Returns the evaluations of all particles across all sub-swarms."""
        return [
            evaluation
            for swarm in self._swarms
            for evaluation in swarm.evaluations
        ]
