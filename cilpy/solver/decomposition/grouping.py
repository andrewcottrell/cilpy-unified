"""Groupings, group splitters and group mergers for cooperative solvers.

A grouping builds a new set of groups from scratch. A splitter (used by
DCPSO) and a merger (used by MCPSO) build the new groups from the
current ones.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional


def shuffled_partition(dimension: int, number_of_groups: int) -> List[List[int]]:
    """
    Shuffles the dimension indices and splits them.

    Args:
        dimension: Total number of decision variables.
        number_of_groups: How many groups to split the shuffled indices into.

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
    """

    def group(self, dimension: int, num_groups: int, **kwargs) -> List[List[int]]:
        """Splits dimensions in order into num_groups groups."""

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
        """Shuffles the dimensions and splits them into num_groups groups."""

        return shuffled_partition(dimension, num_groups)


class RandomGroupSizeGrouping(GroupingStrategy):
    """Random grouping where the group size itself changes each time.

    Every call picks a random group size from the given sizes, so the
    number of groups can change at every regroup.
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

    def group(self, dimension: int, num_groups: Optional[int] = None, **kwargs) -> List[List[int]]:
        """
        Partitions the dimensions using a randomly chosen group size.

        Args:
            dimension: Total number of decision variables.
            num_groups: Ignored, because the number of groups follows from
                the chosen group size.

        Returns:
            A list of groups of dimension indices.
        """

        chosen_group_size = random.choice(self._group_sizes)
        number_of_groups = max(1, dimension // chosen_group_size)

        return shuffled_partition(dimension, number_of_groups)


def cut_into_parts(indices: List[int], number_of_parts: int) -> List[List[int]]:
    """
    Cuts a list of indices in order into the given number of parts.

    When the indices do not divide evenly, the first parts get one extra
    index each, so 5 indices in 2 parts give sizes 3 and 2.

    Args:
        indices: The indices that must be cut.
        number_of_parts: The number of parts to cut into.

    Returns:
        The cut parts as a list of lists containing the indices.
    """

    cut_size = len(indices)//number_of_parts
    remainder = len(indices)%number_of_parts

    parts: List[List[int]] = []
    start = 0

    for part_index in range(number_of_parts):

        size = cut_size
        if part_index < remainder:
            size += 1

        parts.append(indices[start:start + size])
        start += size

    return parts

class GroupSplitter(ABC):
    """Abstract base class for splitting groups.

    Subclasses choose how one group is split by implementing _split_group,
    and can override _select_groups_to_split to choose which groups split.
    """

    def __init__(self, split_factor: int = 2) -> None:
        """
        Initializes the GroupSplitter.

        Args:
            split_factor: Number of parts to split each group into.

        Raises:
            ValueError: If split_factor is less than 2.
        """

        if split_factor < 2:
            raise ValueError(f"split_factor must be at least 2, instead got {split_factor}")

        self.split_factor = split_factor

    def split(self, groups: List[List[int]], state: Optional[dict] = None) -> List[List[int]]:
        """
        Splits the selected groups and keeps the rest unchanged.

        The parts of a group take the place of that group in the list.

        Args:
            groups: The current groups of dimension indices.
            state: Optional coordinator state.

        Returns:
            The new list of groups.
        """

        groups_to_split = self._select_groups_to_split(groups, state)
        new_groups: List[List[int]] = []

        for group_index, group in enumerate(groups):

            if group_index in groups_to_split:
                parts = self._split_group(group, state)
                new_groups.extend(parts)

            else:
                new_groups.append(list(group))

        return new_groups

    def _select_groups_to_split(self, groups: List[List[int]], state: Optional[dict]) -> List[int]:
        """
        Chooses which groups are split. By default every group with more
        than one dimension.

        Args:
            groups: The groups of dimension indices.
            state: Optional coordinator state.

        Returns:
            The indices of the groups to split.
        """

        selected = []

        for group_index, group in enumerate(groups):
            if len(group) > 1:
                selected.append(group_index)

        return selected

    @abstractmethod
    def _split_group(self, group: List[int], state: Optional[dict]) -> List[List[int]]:
        """
        Splits one group into smaller groups.

        Args:
            group: Dimension indices of the group.
            state: Optional coordinator state.

        Returns:
            The parts of the group.
        """


class EvenSplitter(GroupSplitter):
    """A type of GroupSplitter that splits each group into equal parts."""

    def _split_group(self, group: List[int], state: Optional[dict]) -> List[List[int]]:
        """Cuts the group in order into split_factor parts."""

        number_of_parts = min(self.split_factor, len(group))

        return cut_into_parts(list(group), number_of_parts)

class RandomSplitter(GroupSplitter):
    """A GroupSplitter that shuffles each group and then splits it into equal parts."""

    def _split_group(self, group: List[int], state: Optional[dict]) -> List[List[int]]:
        """Shuffles the group and cuts it into split_factor parts."""

        shuffled_group = list(group)
        random.shuffle(shuffled_group)

        number_of_parts = min(self.split_factor, len(shuffled_group))

        return cut_into_parts(shuffled_group, number_of_parts)

class GroupMerger(ABC):
    """Abstract base class for merging groups.

    Subclasses choose which groups are merged together by implementing
    _select_groups_to_merge.
    """

    def __init__(self, merge_factor: int = 2) -> None:
        """
        Initializes the GroupMerger.

        Args:
            merge_factor: Number of groups merged into one.

        Raises:
            ValueError: If merge_factor is less than 2.
        """

        if merge_factor < 2:
            raise ValueError(f"merge_factor must be at least 2 but got {merge_factor}")

        self.merge_factor = merge_factor

    def merge(self, groups: List[List[int]], state: Optional[dict] = None) -> List[List[int]]:
        """
        Merges the groups in each selected block into one group.

        Args:
            groups: The current groups of dimension indices.
            state: Optional coordinator state.

        Returns:
            The new list of groups.
        """

        blocks = self._select_groups_to_merge(groups, state)
        new_groups: List[List[int]] = []

        for block in blocks:

            merged_group = []
            for group_index in block:
                merged_group.extend(groups[group_index])

            new_groups.append(merged_group)

        return new_groups

    @abstractmethod
    def _select_groups_to_merge(self, groups: List[List[int]], state: Optional[dict]) -> List[List[int]]:
        """
        Chooses which groups are merged together.

        Args:
            groups: The groups of dimension indices.
            state: Optional coordinator state.

        Returns:
            Blocks of group indices. Every group appears in exactly one block.
        """

class EvenMerger(GroupMerger):
    """A GroupMerger that merges every merge_factor groups next to each other.

    If the groups do not divide evenly, the last block holds the leftover
    groups, so 5 groups give [1+2], [3+4], [5].
    """

    def _select_groups_to_merge(self, groups: List[List[int]], state: Optional[dict]) -> List[List[int]]:
        """Puts every merge_factor neighbouring groups into one block."""

        blocks = []

        for start in range(0, len(groups), self.merge_factor):

            end = min(start + self.merge_factor, len(groups))
            blocks.append(list(range(start, end)))

        return blocks

class RandomMerger(GroupMerger):
    """A GroupMerger that merges randomly chosen groups.

    The group order is shuffled at every merge, then every merge_factor
    groups are put in one block.
    """

    def _select_groups_to_merge(self, groups: List[List[int]], state: Optional[dict]) -> List[List[int]]:
        """Shuffles the groups and puts every merge_factor of them in one block."""

        group_indices = list(range(len(groups)))
        random.shuffle(group_indices)
        blocks = []

        for start in range(0, len(group_indices), self.merge_factor):
            blocks.append(group_indices[start:start + self.merge_factor])

        return blocks
