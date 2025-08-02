from __future__ import annotations
from local_information.lattice.lattice_dict import LatticeDict
from functools import cached_property
import numpy as np

from local_information.core.petz_map import ptrace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_information.typedefs import LatticeDictKeyTuple


class StateBoundary:
    """
    Class to handle the boundaries of asymptotic translational invariant systems.

    Important note: Here we assume the boundary terms to be time-invariant under
    evolution with the system operator! This will change in the future version.
    In future versions, this class will be used as a data structure
    and updated during evolution.
    """

    def __init__(
        self,
        boundary_left: LatticeDict,
        boundary_right: LatticeDict,
    ):
        self._boundary_left = boundary_left
        self._boundary_right = boundary_right

    @classmethod
    def from_lattice_dict(cls, density_matrix: LatticeDict):
        level = density_matrix.get_max_level()
        left_boundary_key = (density_matrix.smallest_at_level(level), level)
        right_boundary_key = (density_matrix.largest_at_level(level), level)

        boundary_left = LatticeDict.from_dict(
            {left_boundary_key: density_matrix[left_boundary_key]}
        )
        boundary_right = LatticeDict.from_dict(
            {right_boundary_key: density_matrix[right_boundary_key]}
        )
        return cls(
            boundary_left=boundary_left,
            boundary_right=boundary_right,
        )

    @property
    def boundary_key_left(self) -> LatticeDictKeyTuple:
        """Get the boundary key on the left end of the state."""
        n_min = self._boundary_left.smallest_at_level(self.level)
        return n_min, self.level

    @property
    def boundary_key_right(self) -> LatticeDictKeyTuple:
        """Get the boundary key on the left end of the state."""
        n_max = self._boundary_right.largest_at_level(self.level)
        return n_max, self.level

    @property
    def level(self):
        level_left = self._boundary_left.get_max_level()
        level_right = self._boundary_right.get_max_level()
        assert level_left == level_right
        return level_left

    @cached_property
    def lowest_level_left(self):
        """Get the lowest level density matrix on the left end of the state."""
        return ptrace(
            self._boundary_left[self.boundary_key_left],
            spins_to_trace_out=self.level,
            end="right",
        )

    @cached_property
    def lowest_level_right(self):
        """Get the lowest level density matrix on the right end of the state."""
        return ptrace(
            self._boundary_right[self.boundary_key_right],
            spins_to_trace_out=self.level,
            end="left",
        )

    def update_boundary_keys_right(self, key, boundary: np.ndarray | None = None):
        boundary_right = self._boundary_right[self.boundary_key_right]
        old_boundary_key = self.boundary_key_right
        if boundary:
            self._boundary_right[key] = boundary
        else:
            self._boundary_right[key] = boundary_right
        self._boundary_right.pop(old_boundary_key)

    def update_boundary_keys_left(self, key, boundary: np.ndarray | None = None):
        boundary_left = self._boundary_left[self.boundary_key_left]
        old_boundary_key = self.boundary_key_left
        if boundary:
            self._boundary_left[key] = boundary
        else:
            self._boundary_left[key] = boundary_left
        self._boundary_left.pop(old_boundary_key)
