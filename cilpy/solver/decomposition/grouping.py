"""Grouping strategies for cooperative decomposition solvers."""

import random
from abc import ABC, abstractmethod
from typing import List


class GroupingStrategy(ABC):
    """Abstract base for dimension grouping strategies.

    A grouping strategy partitions the indices into
    `num_groups` subsets. The coordinator calls
    `group()` once at initialisation and again whenever a regroup event needs to happen.
    """

    @abstractmethod
    def group(self, dimension: int, num_groups: int, **kwargs) -> List[List[int]]:
        """Partition ``[0, dimension)`` into ``num_groups`` groups.

        Args:
            dimension: Total number of decision variables.
            num_groups: Desired number of groups.
            **kwargs: Strategy-specific keyword arguments

        Returns:
            A list of ``num_groups`` lists of integer indices.
        """


class StaticEvenGrouping(GroupingStrategy):
    """Partitions dimensions into equal-sized slices.

    Indices are assigned left-to-right in order. The last group takes any
    remainder so that all indices are covered even when ``dimension`` is not
    evenly divisible by ``num_groups``.

    Example:
        >>> g = StaticEvenGrouping()
        >>> g.group(10, 3)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    """

    def group(self, dimension: int, num_groups: int, **kwargs) -> List[List[int]]:
        
        base = dimension // num_groups
        remainder = dimension % num_groups
        groups: List[List[int]] = []

        start = 0
        for i in range(num_groups):
            # Assign the remainder if this is the last group
            is_last_group = i == num_groups - 1
            size = base + (remainder if is_last_group else 0)

            groups.append(list(range(start, start + size)))
            start += size

        return groups


class RandomGrouping(GroupingStrategy):
    """Partitions dimensions by shuffling indices then splitting into slices.

    The last group absorbs any remainder.

    Example:
        >>> import random;
        >>> g = RandomGrouping()
        >>> g.group(6, 2)
        [[4, 0, 2], [5, 1, 3]]
    """
    def group(self, dimension: int, num_groups: int, **kwargs) -> List[List[int]]:
        indices = list(range(dimension))
        random.shuffle(indices)

        base = dimension // num_groups
        remainder = dimension % num_groups
        groups: List[List[int]] = []

        start = 0
        for i in range(num_groups):
            is_last_group = i == num_groups - 1
            size = base + (remainder if is_last_group else 0)
            groups.append(indices[start:start + size])
            start += size

        return groups
