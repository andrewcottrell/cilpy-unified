# cilpy/solver/decomposition/trigger.py
"""Regroup triggers for cooperative decomposition solvers."""

from abc import ABC, abstractmethod

class RegroupTrigger(ABC):


    @abstractmethod
    def should_trigger(self, state: dict) -> bool:
        """Decides whether a regroup event should happen now.

        Args:
            state: Is the coordinator state dictionary. It holds the key
                'iteration'.

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


class FixedIntervalTrigger(RegroupTrigger):


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


class NeverTrigger(RegroupTrigger):
    """This never triggers. Used by solvers with a static grouping."""

    def should_trigger(self, state: dict) -> bool:
        return False
