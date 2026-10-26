# cilpy/solver/decomposition/trigger.py
"""Regroup triggers for cooperative decomposition solvers."""

import math
from typing import Optional
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
        """Returns True when a trigger should fire."""

        iteration = state["iteration"]

        has_started = iteration > 0
        is_interval_boundary = iteration % self._interval == 0

        return has_started and is_interval_boundary


class EvaluationIntervalTrigger(RegroupTrigger):
    """Triggers every fixed amount of fitness evaluations."""

    def __init__(self, interval: int) -> None:
        """Initializes the EvaluationIntervalTrigger."""

        if interval < 1:
            raise ValueError(f"interval must be at least 1, but got {interval}")

        self._interval = interval
        self._next_threshold = interval


    def should_trigger(self, state: dict) -> bool:
        """Returns True once the evaluations used reach the next threshold."""

        return state["evaluations"] >= self._next_threshold

    def on_triggered(self, state: dict) -> None:
        """Moves the threshold one interval further."""


        self._next_threshold += self._interval

    def reset(self) -> None:
        """Sets the threshold back to the first interval."""
        self._next_threshold = self._interval


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
        """Returns True after patience iterations in a row without a meaningful improvement."""

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
        """Forgets the best fitness and the patience count of the previous run."""
        self._best_fitness_so_far = None
        self._iterations_without_improvement = 0


class NeverTrigger(RegroupTrigger):
    """This never triggers. Used by solvers with a static grouping."""

    def should_trigger(self, state: dict) -> bool:
        """Always returns False."""
        return False


def restructure_interval(total_budget: int, dimension: int, factor: int) -> int:
    """Calculates how much of the budget should be between regroup events.

    Args:
        total_budget: The total budget of the run in iterations or in
            fitness evaluations.
        dimension: Total number of decision variables.
        factor: Amount of parts a group splits into.

    Returns:
        The interval between regroup events in the same unit as total_budget.

    Raises:
        ValueError: if total_budget or dimension is below 1 or factor is below 2.
    """

    if total_budget < 1:
        raise ValueError(f"total_budget must be at least 1 but is {total_budget}")

    if dimension < 1:
        raise ValueError(f"dimension must be at least 1 but is {dimension}")

    if factor < 2:
        raise ValueError(f"factor must be at least 2 but is {factor}")

    number_of_restructures = math.log(dimension) / math.log(factor)
    number_of_phases = 1 + number_of_restructures

    interval = math.floor(round(total_budget / number_of_phases, 6))

    return max(1, interval)


def restructure_trigger(dimension: int, factor: int, max_iterations: Optional[int] = None,
                        max_evaluations: Optional[int] = None) -> RegroupTrigger:
    """Creates a trigger that spaces the restructures evenly over the budget.

    Args:
        dimension: number of decision variables
        factor: Amount of parts a group splits into
        max_iterations: The iteration budget of the run.
        max_evaluations: The fitness evaluation budget of the run.

    Returns:
        An EvaluationIntervalTrigger for an evaluation budget
    """

    if max_evaluations is not None:
        
        interval = restructure_interval(max_evaluations, dimension, factor)
        return EvaluationIntervalTrigger(interval=interval)

    interval = restructure_interval(max_iterations, dimension, factor)

    return FixedIntervalTrigger(interval=interval)
