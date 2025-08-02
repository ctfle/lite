from __future__ import annotations
from typing import TYPE_CHECKING
from functools import cached_property
import logging

import numpy as np

from local_information.lattice.lattice_dict import LatticeDict

if TYPE_CHECKING:
    from local_information.typedefs import LatticeDictKeyTuple

from local_information.mpi.mpi_funcs import get_mpi_variables

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class Distributor:
    """
    Converts a LatticeDict into a np.ndarray for scattering. Assumes that all entries of
    the LatticeDict have the same shape. Must take responsibility to distribute tasks
    input should be single lattice dict that is converted into single np.ndarray (which is scattered)

    Pretty sure we can get away without the n_min, n_max values.
    Remove them and just use the dict directly.
    The implementation gets easier as well

    Note: I think it might be smarter to use scatter and gather here since
    its a bit of a hustle to create a single np.array (it has to be contiguous in mem
    which means we need to copy things around). This very likely eats up the advantage we might have
    using Scatter and Gather (instead of scatter and gather). Good news. We can keep quite some of the old
    local_information (improve it).

    """

    def __init__(self, density_matrices: LatticeDict, level: int):
        assert density_matrices, "Empty LatticeDict not allowed"
        if RANK == 0:
            # this should be checked only on root which holds the data of interest
            assert list(density_matrices.keys_at_level(level)), (
                f"No density matrices at level {level}"
            )
        self._density_matrices = density_matrices
        self._level = level

    @cached_property
    def n_min(self) -> float:
        return self._density_matrices.smallest_at_level(self._level)

    @cached_property
    def n_max(self) -> float:
        return self._density_matrices.largest_at_level(self._level)

    @cached_property
    def _number_of_entries(self) -> int:
        return len(list(self._density_matrices.keys_at_level(self._level)))

    @cached_property
    def _shape(self) -> tuple[int, int]:
        return next(self._density_matrices.values_at_level(self._level)).shape

    @property
    def _ordered_keys(self) -> list[list[LatticeDictKeyTuple]]:
        return [key for key in self._density_matrices.keys_at_level(self._level)]

    @cached_property
    def _ordered_values_as_array(self) -> np.ndarray:
        ordered_values = np.empty(
            shape=(self._number_of_entries, *self._shape), dtype="float64"
        )
        for ind, density_matrix in enumerate(
            self._density_matrices.values_at_level(self._level)
        ):
            ordered_values[ind] = density_matrix

        return ordered_values

    def distribute(
        self, number_of_workers: int = SIZE, shift: int = 1
    ) -> list[LatticeDict] | None:
        self._check_shift(shift)

        if RANK == 0:
            split_up_lattice = self._split_lattice(
                number_of_splits=number_of_workers, shift=shift
            )
        else:
            split_up_lattice = None

        return split_up_lattice

    def _split_lattice(
        self, number_of_splits: int, shift: int = 1
    ) -> list[LatticeDict]:
        """
        Split the input lattice of density matrices in several sub-lattices.
        `shift` determines the overlap of the keys at a given level.
        """

        sub_lattices = []

        split_up_keys = self._split_keys(number_of_splits=number_of_splits, shift=shift)
        for key_block in split_up_keys:
            lattice_block = LatticeDict()
            for key in key_block:
                lattice_block[key] = self._density_matrices[key]

            sub_lattices.append(lattice_block)

        return sub_lattices

    def _split_keys(self, number_of_splits: int, shift: int = 1) -> list[list[tuple]]:
        """Split the keys in blocks"""
        split_up_keys = np.array_split(self._ordered_keys, number_of_splits)
        # get the keys as list of tuples
        split_up_keys = list(map(lambda x: list([tuple(y) for y in x]), split_up_keys))

        # if shifting is required
        self._add_keys_of_next_block(split_up_keys=split_up_keys, shift=shift)
        return split_up_keys

    @staticmethod
    def _add_keys_of_next_block(split_up_keys: list[list[tuple]], shift: int = 1):
        """
        Add the first `shift` many elements of the keys
        in the consecutive block to each block.
        """
        for k, key_block in enumerate(split_up_keys[:-1]):
            next_key_block = split_up_keys[k + 1]
            for s in range(shift):
                if s < len(next_key_block):
                    key_block.append(next_key_block[s])

    def scatter(self, number_of_workers: int = SIZE, shift: int = 1) -> LatticeDict:
        # the return values of `distribute` are None for all but RANK == 0
        split_up_lattice = self.distribute(
            number_of_workers=number_of_workers, shift=shift
        )
        return COMM.scatter(split_up_lattice, root=0)

    @staticmethod
    def gather(data: LatticeDict) -> LatticeDict | None:
        """Gather data at root. Returns None on all other processes."""
        gathered_data = COMM.gather(data, root=0)

        if RANK == 0:
            joint_lattice = gathered_data[0]
            for lattice in gathered_data:
                joint_lattice.merge(lattice)
        else:
            joint_lattice = None

        return joint_lattice

    def _check_shift(self, shift: int):
        assert shift == 0 or shift == 1, (
            "shift values other than 0 or 1 are not supported."
        )
