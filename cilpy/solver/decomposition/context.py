# cilpy/solver/decomposition/context.py
from typing import List, Optional

class ContextVector:
    """
    A shared D-dimensional state updated cooperatively by all sub-swarms.

    Each sub-swarm owns one group of dimension indices. After every
    iteration the sub-swarm writes its best partial solution back into
    this vector using the update_group function. Any sub-swarm that needs a full
    D-dimensional solution to compute fitness calls the get_full_vector function,
    which returns a copy so that each sub-swarm gets an independent
    snapshot to write into without corrupting the shared state.
    """

    def __init__(
        self,
        dimension: int,
        groups: List[List[int]],
        initial_vector: Optional[List[float]] = None,
    ) -> None:
        """
        Initializes the ContextVector.

        Args:
            dimension: Total number of decision variables (D).
            groups: A list of N lists of global indices.
            initial_vector: Optional starting values for the internal vector.
                Defaults to zeros when not provided.
        """
        self._validate_groups(dimension, groups)

        if initial_vector is not None:
            self._full_vector: List[float] = list(initial_vector)
        else:
            self._full_vector: List[float] = [0.0] * dimension

        self._groups: List[List[int]] = [list(g) for g in groups]

    def update_group(self, group_idx: int, sub_solution: List[float]) -> None:
        """
        Writes a sub-swarm's best partial solution into the context vector.

        Args:
            group_idx: Index of the group whose values are being updated.
            sub_solution: Partial solution of length len(groups[group_idx]).
        """
        for k, global_idx in enumerate(self._groups[group_idx]):
            self._full_vector[global_idx] = sub_solution[k]

    def get_full_vector(self) -> List[float]:
        """
        Returns a copy of the current assembled D-dimensional solution.

        Gives an independent copy so groups can safely overwrite
        their group's positions without affecting the shared state or
        other sub-swarms.

        Returns:
            An independant copy of the internal full vector.
        """
        return list(self._full_vector)


    def reassign_groups(self, new_groups: List[List[int]]) -> None:
        """
        Updates group ownership without discarding accumulated values.

        Called during regroup events.
        Only the ownership mapping changes and the values already stored at
        every global index are preserved so that accumulated search
        knowledge is not lost.

        Args:
            new_groups: New partition of range(dimension) into groups.

        Raises:
            ValueError: If new_groups does not partition the same set
                of indices as the current groups.
        """
        dimension = len(self._full_vector)
        self._validate_groups(dimension, new_groups)
        self._groups = [list(g) for g in new_groups]


    @staticmethod
    def _validate_groups(dimension: int, groups: List[List[int]]) -> None:
        """
        Checks if groups do not form a valid partition.

        Args:
            dimension: Expected total number of indices.
            groups: Candidate partition to validate.

        Raises:
            ValueError: If any index is missing, duplicated, or out of range.
        """
        all_indices = []
        for group in groups:
            for idx in group:
                all_indices.append(idx)
                
        if sorted(all_indices) != list(range(dimension)):
            raise ValueError(
                f"groups must partition range({dimension}) exactly once, "
                f"but got indices: {sorted(all_indices)}"
            )
