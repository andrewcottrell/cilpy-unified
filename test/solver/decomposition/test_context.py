# test/solver/decomposition/test_context.py
"""
Unit tests for ContextVector.
"""

from cilpy.solver.decomposition.context import ContextVector

class TestUpdateGroup:
    """Tests for writing a group's values into the context."""

    def test_values_land_at_the_correct_positions(self):
        """Updating group 0 writes its values at positions 0 and 1."""

        context = make_two_group_context()

        context.update_group(group_idx=0, sub_solution=[10.0, 20.0])
        full_vector = context.get_full_vector()

        assert full_vector[0] == 10.0
        assert full_vector[1] == 20.0

    def test_other_group_positions_are_unchanged(self):
        """Updating group 0 must not touch positions owned by group 1."""
        context = make_two_group_context()

        context.update_group(group_idx=0, sub_solution=[10.0, 20.0])
        full_vector = context.get_full_vector()

        assert full_vector[2] == 0.0
        assert full_vector[3] == 0.0

    def test_updating_second_group_writes_its_positions(self):
        """Updating group 1 writes its values at positions 2 and 3."""
        context = make_two_group_context()

        context.update_group(group_idx=1, sub_solution=[7.0, 8.0])
        full_vector = context.get_full_vector()

        assert full_vector[2] == 7.0
        assert full_vector[3] == 8.0

    def test_updating_a_group_twice_keeps_the_latest_values(self):
        """A second update to a group must overwrite the first."""
        context = make_two_group_context()

        context.update_group(group_idx=0, sub_solution=[1.0, 1.0])
        context.update_group(group_idx=0, sub_solution=[9.0, 9.0])
        full_vector = context.get_full_vector()

        assert full_vector[0] == 9.0
        assert full_vector[1] == 9.0

    def test_updating_both_groups_assembles_the_full_vector(self):
        """Updating both groups gives the complete vector in the right order."""
        context = make_two_group_context()

        context.update_group(group_idx=0, sub_solution=[1.0, 2.0])
        context.update_group(group_idx=1, sub_solution=[3.0, 4.0])

        assert context.get_full_vector() == [1.0, 2.0, 3.0, 4.0]

class TestGetFullVector:
    """Tests for reading the full vector from the context."""

    def test_returns_a_list(self):
        """The full vector is returned as a list."""
        context = make_two_group_context()
        assert isinstance(context.get_full_vector(), list)

    def test_returned_list_has_correct_length(self):
        """The full vector has one value per dimension."""
        context = make_two_group_context()
        assert len(context.get_full_vector()) == 4

    def test_mutating_returned_list_does_not_affect_context(self):
        """The returned list is a copy, so changing it leaves the context unchanged."""

        context = make_two_group_context()
        context.update_group(group_idx=0, sub_solution=[5.0, 6.0])

        returned_copy = context.get_full_vector()
        returned_copy[0] = 999.0

        fresh_copy = context.get_full_vector()
        assert fresh_copy[0] == 5.0

    def test_initial_vector_is_used_when_provided(self):
        """The context starts with the given initial vector."""

        initial = [1.0, 2.0, 3.0, 4.0]
        groups = [[0, 1], [2, 3]]
        context = ContextVector(dimension=4, groups=groups, initial_vector=initial)

        assert context.get_full_vector() == [1.0, 2.0, 3.0, 4.0]


class TestReassignGroups:
    """Tests for changing which group owns which positions."""

    def test_existing_values_are_preserved_after_reassign(self):
        """
        Every value written before the regroup should still be at the same global index.
        """
        groups = [[0, 1], [2, 3]]
        context = ContextVector(dimension=4, groups=groups)
        context.update_group(group_idx=0, sub_solution=[1.0, 2.0])
        context.update_group(group_idx=1, sub_solution=[3.0, 4.0])

        new_groups = [[0, 2], [1, 3]]
        context.reassign_groups(new_groups)

        assert context.get_full_vector() == [1.0, 2.0, 3.0, 4.0]

    def test_subsequent_update_uses_new_group_indices(self):
        """
        After reassigning update_group should write to the new group's indices.
        New group 0 owns [0, 2] so updating it should change positions 0 and 2.
        """
        groups = [[0, 1], [2, 3]]
        context = ContextVector(dimension=4, groups=groups)
        context.update_group(group_idx=0, sub_solution=[1.0, 2.0])
        context.update_group(group_idx=1, sub_solution=[3.0, 4.0])

        context.reassign_groups([[0, 2], [1, 3]])
        context.update_group(group_idx=0, sub_solution=[10.0, 20.0])

        full_vector = context.get_full_vector()
        assert full_vector[0] == 10.0
        assert full_vector[2] == 20.0
        assert full_vector[1] == 2.0
        assert full_vector[3] == 4.0

def make_two_group_context():
    """Returns a 4-D context of zeros with groups [0, 1] and [2, 3]."""
    groups = [[0, 1], [2, 3]]
    return ContextVector(dimension=4, groups=groups)
