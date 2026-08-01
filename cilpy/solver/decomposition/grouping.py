"""Grouping strategies for cooperative decomposition solvers."""

import random
from abc import ABC, abstractmethod
from typing import List, Optional


def shuffled_partition(dimension: int, number_of_groups: int) -> List[List[int]]:
    """
    Shuffles the dimension indices and splits them.

    Args:
        dimension: Total number of decision variables.
        number_of_groups: How many groups we to split the shuffled indices into.

    Returns:
        A list of `number_of_groups` lists of global dimension indices. The last
        group takes any remainder.
    """

    indices = list(range(dimension))
    random.shuffle(indices)
    base_size = dimension // number_of_groups
    remainder = dimension % number_of_groups

    groups: List[List[int]] = []
    start = 0

    for group_index in range(number_of_groups):
        
        is_last_group = group_index == number_of_groups - 1
        size = base_size + (remainder if is_last_group else 0)

        groups.append(indices[start:start + size])
        start += size

    return groups


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

        return shuffled_partition(dimension, num_groups)


class RandomGroupSizeGrouping(GroupingStrategy):
    """Random grouping where the group size it self is changed each time.
    """

    def __init__(self, group_sizes: List[int]) -> None:
        """
        Initializes the RandomGroupSizeGrouping.

        Args:
            group_sizes: The set of group sizes to choose from.
                For example {2, 5, 10, 50, 100} for
                100 dimensions.

        Raises:
            ValueError: If group_sizes is empty or holds a size below 1.
        """

        if not group_sizes:
            raise ValueError("group_sizes must hold at least one size")

        for size in group_sizes:
            if size < 1:
                raise ValueError(
                    f"every group size must be at least 1, but got {size}")

        self._group_sizes = list(group_sizes)

    def group(self, dimension: int, num_groups: Optional[int] = None,
              **kwargs) -> List[List[int]]:
        """
        Partitions the dimensions using a randomly chosen group size.

        Args:
            dimension: Total number of decision variables.

        Returns:
            A list of groups of dimension indices.
        """

        chosen_group_size = random.choice(self._group_sizes)
        number_of_groups = max(1, dimension // chosen_group_size)

        return shuffled_partition(dimension, number_of_groups)
