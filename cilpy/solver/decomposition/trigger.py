# cilpy/solver/decomposition/trigger.py
"""Regroup triggers for cooperative decomposition solvers."""

import math
from abc import ABC, abstractmethod

class RegroupTrigger(ABC):
    """Abstract base for regroup triggers.

    A regroup trigger chooses when the coordinator should change
    current dimension grouping and use a new one.

    """

    @abstractmethod
    def should_trigger(self, state: dict) -> bool:
        """Decides whether a regroup event should happen now.

        Args:
            state: Is the coordinator state dictionary. It holds the keys
                iteration and best_fitness.

        Returns:
            True when the coordinator should regroup and False otherwise.
        """

    def on_triggered(self, state: dict) -> None:
        """A hook called immediately after a regroup fires.

        Used for triggers that keep
        internal counters override this to reset them.

        Args:
            state: The same state dictionary passed to should_trigger.
        """

    def reset(self) -> None:
        """Clears the state carried over from a previous run."""


class FixedIntervalTrigger(RegroupTrigger):
    """Triggers every fixed amount of iterations. Never fires at
        iteration zero.
    """

    def __init__(self, interval: int) -> None:
        """
        Initializes the FixedIntervalTrigger.

        Args:
            interval: Number of iterations between regroup events.
                This must be at least 1.

        Raises:
            ValueError: If interval is less than 1.
        """

        if interval < 1:
            raise ValueError(f"interval must be at least 1, but got {interval}")

        self._interval = interval

    def should_trigger(self, state: dict) -> bool:

        iteration = state["iteration"]

        has_started = iteration > 0
        is_interval_boundary = iteration % self._interval == 0

        return has_started and is_interval_boundary


class StagnationTrigger(RegroupTrigger):
    """This triggers when the search stops making good progress and begins to stagnate. 
    """

    def __init__(self, patience: int, tolerance: float = 1e-4) -> None:
        """
        Initializes the StagnationTrigger.

        Args:
            patience: Number of iterations without good
                improvement that is alowed before a regroup is triggered.

            tolerance: The amount of improvement that is seen as meaningfull.
                This amount is relative towards the fitness.

        Raises:
            ValueError: If patience is less than 1 or tolerance is negative.
        """

        if patience < 1:
            raise ValueError(f"patience must be at least 1, but got {patience}")

        if tolerance < 0:
            raise ValueError(f"tolerance must not be negative, but got {tolerance}")


        self._patience = patience
        self._tolerance = tolerance

        self._best_fitness_so_far = None
        self._iterations_without_improvement = 0

    def _is_meaningful_improvement(self, current_best_fitness: float) -> bool:
        """
        Decides whether the current fitness improved meaningfull enough or not.

        Args:
            current_best_fitness: The coordinators best assembled fitness so far.

        Returns:
            True if the improvement is at least the required fraction.
        """
    
        reference_magnitude = abs(self._best_fitness_so_far)

        if not math.isfinite(reference_magnitude):
            return math.isfinite(current_best_fitness)

        improvement = self._best_fitness_so_far - current_best_fitness
        required_improvement = self._tolerance * reference_magnitude

        return improvement > required_improvement


    def should_trigger(self, state: dict) -> bool:

        current_best_fitness = state["best_fitness"]

        if self._best_fitness_so_far is None:
            self._best_fitness_so_far = current_best_fitness
            self._iterations_without_improvement = 0
            return False

        if self._is_meaningful_improvement(current_best_fitness):

            self._best_fitness_so_far = current_best_fitness
            self._iterations_without_improvement = 0

        else:
            self._iterations_without_improvement += 1

        return self._iterations_without_improvement >= self._patience

    def on_triggered(self, state: dict) -> None:
        """Resets the patience after a regroup has been performed.

        Args:
            state: The same state dictionary.
        """
        self._iterations_without_improvement = 0

    def reset(self) -> None:
        self._best_fitness_so_far = None
        self._iterations_without_improvement = 0


class NeverTrigger(RegroupTrigger):
    """This never triggers. Used by solvers with a static grouping."""

    def should_trigger(self, state: dict) -> bool:
        return False
