# test/solver/decomposition/test_grouping.py
"""
Unit tests for StaticEvenGrouping and RandomGrouping.
"""

import random
from cilpy.solver.decomposition.grouping import RandomGrouping, StaticEvenGrouping

class TestStaticEvenGrouping:
    """Tests for the in-order grouping."""

    def test_even_split_produces_equal_contiguous_slices(self):
        """9 dimensions in 3 groups give three in-order groups of 3."""

        grouping = StaticEvenGrouping()

        groups = grouping.group(dimension=9, num_groups=3)

        assert groups == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_remainder_goes_to_the_last_group(self):
        """The leftover dimension goes to the last group."""

        grouping = StaticEvenGrouping()

        groups = grouping.group(dimension=10, num_groups=3)

        assert groups[0] == [0, 1, 2]
        assert groups[1] == [3, 4, 5]
        assert groups[2] == [6, 7, 8, 9]

    def test_output_is_a_valid_partition(self):
        """Every dimension appears in exactly one group."""

        grouping = StaticEvenGrouping()

        groups = grouping.group(dimension=12, num_groups=4)

        assert_valid_partition(groups, dimension=12, expected_num_groups=4)

    def test_single_group_receives_all_indices(self):
        """With one group, that group holds every dimension."""

        grouping = StaticEvenGrouping()

        groups = grouping.group(dimension=5, num_groups=1)

        assert groups == [[0, 1, 2, 3, 4]]

    def test_same_call_always_returns_the_same_result(self):
        """The grouping is not random."""

        grouping = StaticEvenGrouping()

        first_result = grouping.group(dimension=15, num_groups=3)
        second_result = grouping.group(dimension=15, num_groups=3)

        assert first_result == second_result

class TestRandomGrouping:
    """Tests for the shuffled grouping."""

    def test_output_is_a_valid_partition(self):
        """Every dimension appears in exactly one group."""

        random.seed(42)
        grouping = RandomGrouping()

        groups = grouping.group(dimension=12, num_groups=4)

        assert_valid_partition(groups, dimension=12, expected_num_groups=4)

    def test_remainder_goes_to_the_last_group(self):
        """The leftover dimension goes to the last group."""

        random.seed(0)
        grouping = RandomGrouping()

        groups = grouping.group(dimension=10, num_groups=3)

        group_sizes = [len(group) for group in groups]
        assert group_sizes[0] == 3
        assert group_sizes[1] == 3
        assert group_sizes[2] == 4

    def test_different_seeds_produce_different_orderings(self):
        """Different random seeds give different groups."""

        grouping = RandomGrouping()

        random.seed(0)
        groups_with_seed_zero = grouping.group(dimension=20, num_groups=4)

        random.seed(1)
        groups_with_seed_one = grouping.group(dimension=20, num_groups=4)

        assert groups_with_seed_zero != groups_with_seed_one

    def test_same_seed_produces_the_same_result(self):
        """The same random seed gives the same groups."""

        grouping = RandomGrouping()

        random.seed(99)
        first_result = grouping.group(dimension=15, num_groups=3)

        random.seed(99)
        second_result = grouping.group(dimension=15, num_groups=3)

        assert first_result == second_result

# --- Helper functions ---

def assert_valid_partition(groups, dimension, expected_num_groups):
    """Checks the number of groups and that every index appears exactly once."""

    assert len(groups) == expected_num_groups

    all_indices = []
    for group in groups:
        all_indices.extend(group)
    assert sorted(all_indices) == list(range(dimension))
