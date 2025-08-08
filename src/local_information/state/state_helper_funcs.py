from __future__ import annotations

import logging
from copy import deepcopy
import numpy as np
from local_information.core.utils import get_higher_level_single_processing
from local_information.lattice.lattice_dict import LatticeDict
from local_information.core.utils import compute_mutual_information_at_level
from local_information.mpi.mpi_funcs import get_mpi_variables
from typing import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_information.typedefs import LatticeDictKeyTuple

logger = logging.getLogger()

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


def total_information(rho_dict: LatticeDict, dyn_max_l: int, level: int) -> float:
    """Computes the total mutual information summed over all LatticeDict sites"""
    lowest_level_dict = deepcopy(rho_dict)
    info = LatticeDict()
    sum_info = 0.0

    # bcast `dyn_max_l` and `level` to ensure loop is run on all workers
    dyn_max_l = COMM.bcast(dyn_max_l, root=0)
    level = COMM.bcast(level, root=0)
    for j in range(dyn_max_l - level):
        lowest_level_dict, _ = compute_mutual_information_at_level(
            lowest_level_dict, dyn_max_l - j
        )
        lowest_level_dict = COMM.bcast(lowest_level_dict, root=0)

        _, lower_level_info = compute_mutual_information_at_level(
            lowest_level_dict, dyn_max_l - j - 1
        )
        lower_level_info = COMM.bcast(lower_level_info, root=0)
        info += lower_level_info
        sum_info += sum(info.values_at_level(dyn_max_l - j - 1))

    return sum_info


def check_density_matrix_sequence(density_matrix_sequence: Sequence[np.ndarray]):
    """! Check if initial density matrices given as a list are proper density matrices"""
    for rho in density_matrix_sequence:
        # check if it's a square matrix
        if not len(set(rho.shape)) == 1:
            raise ValueError(
                "input incorrect: initial density matrix is not a square matrix"
            )

        # check if it has trace one
        if not np.isclose(np.trace(rho), 1.0, atol=1e-8):
            raise ValueError(
                "input incorrect: initial density matrix has not unit trace"
            )

        # check if it's Hermitian
        rho_H = rho.transpose().conjugate()
        if not np.allclose(rho, rho_H, atol=1e-8):
            raise ValueError("input incorrect: initial density matrix is not Hermitian")

        # check if it's semi-positive definite
        e, v = np.linalg.eigh(rho)
        if e.any() <= 0.0:
            raise ValueError(
                "input incorrect: initial density matrix is not semi-positive definite"
            )
    pass


def check_level_overhead(
    density_matrices_sequence: Sequence[np.ndarray], level_overhead: int
):
    """
    Checks that the given level-overhead is implementable
    """
    sequence_length = len(density_matrices_sequence)
    input_dim = get_base_2_dim(density_matrices_sequence[0])
    max_possible_level = input_dim * sequence_length - 1
    if input_dim + level_overhead - 1 > max_possible_level:
        raise ValueError(
            "Provided level_overhead to big for given sequence of density matrices"
        )


def get_base_2_dim(matrix: np.ndarray) -> int:
    return int(np.log(len(matrix)) / np.log(2))


def get_largest_dim(matrix_sequence: Sequence[np.ndarray]) -> int:
    if matrix_sequence:
        return max(map(get_base_2_dim, matrix_sequence))
    else:
        return 0


def add_higher_level_site(
    input_lattice: LatticeDict, key: LatticeDictKeyTuple, next_key: LatticeDictKeyTuple
):
    assert key[1] == next_key[1], "keys must be associated with same level"
    level = key[1]
    temp_dict = LatticeDict()
    temp_dict[key] = input_lattice[key]
    temp_dict[next_key] = input_lattice[next_key]

    return get_higher_level_single_processing(temp_dict, level)
