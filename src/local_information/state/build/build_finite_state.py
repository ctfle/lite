from __future__ import annotations

import logging
import numpy as np
from local_information.core.utils import get_higher_level_single_processing
from local_information.lattice.lattice_dict import LatticeDict
from local_information.state.state_helper_funcs import get_base_2_dim
from local_information.core.utils import (
    compute_lower_level,
    compute_lower_level_sparse,
)
from local_information.core.petz_map import PetzMap
from local_information.mpi.mpi_funcs import get_mpi_variables
from typing import Sequence

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


def get_finite_state(
    density_matrix_sequence: Sequence[np.ndarray], max_l: int
) -> tuple[LatticeDict[tuple[float, int], np.ndarray], int]:
    """
    Computes all the density matrices at level max_l if possible.
    If not, builds the largest possible level. This can happen if the
    given sequence is not long enough.
    """
    assert density_matrix_sequence, (
        "density_matrix_sequence must contain at least one element"
    )
    check_equal_dims(density_matrix_sequence)

    input_dim = get_base_2_dim(density_matrix_sequence[0])
    input_level = input_dim - 1
    if len(density_matrix_sequence) == 1:
        # get size of structure_part
        state_level = input_level
        state_dict = LatticeDict()

        if input_level > max_l:
            logger.info(
                f"density_matrix_sequence has level {input_dim} larger than max_l={max_l}"
            )
            # reduce to level max_l
            state_dict[(input_level / 2, input_level)] = density_matrix_sequence[0]
            ell = input_level
            while ell > max_l:
                state_dict = compute_lower_level(state_dict, ell)
                ell -= 1
            state_dict.kill_all_except(max_l)
            state_level = max_l
        else:
            state_dict[(input_level / 2, input_level)] = density_matrix_sequence[0]

    else:
        if input_dim * len(density_matrix_sequence) - 1 <= max_l:
            # compute the highest level density matrix available from the input
            (
                state_dict,
                state_level,
            ) = build_higher_level_from_sequence_of_density_matrices(
                density_matrix_sequence, max_l
            )
        else:
            # the top of the structure_part triangle has level
            # which is larger than max_l: stop at max_l
            state_dict = get_density_matrices_at_level(
                density_matrix_sequence, level=max_l
            )
            state_level = max_l

    return state_dict, state_level


def check_equal_dims(density_matrix_sequence: Sequence[np.ndarray]):
    check_len = set(map(len, density_matrix_sequence))
    if not len(check_len) == 1:
        raise AssertionError(
            "input matrices for structure_part must be of the same size"
        )


def build_higher_level_from_sequence_of_density_matrices(
    density_matrix_sequence: Sequence[np.ndarray], level: int
) -> tuple[LatticeDict, int]:
    """
    Builds higher level density matrices using the Petz map.
    Here we assume that the input density matrices do not overlap
    i.e. they are associated with non-overlapping subsystems
    and are all of equal dimension.
    We increase the level until we arrive at a level >= required
    from the input given by `level`.
    """
    sequence_length = len(density_matrix_sequence)
    assert sequence_length >= 1, (
        "Sequence of density matrices must contain at least one element."
    )
    input_dim = get_base_2_dim(density_matrix_sequence[0])
    ell = input_dim - 1

    keys = get_non_overlaping_keys(ell, sequence_length)
    higher_level_density_matrices = LatticeDict.from_list(
        keys=keys, values=density_matrix_sequence
    )
    while len(keys) > 1 and ell < level:
        updated_keys = []
        for i, (key_A, key_B) in enumerate(zip(keys[:-1], keys[1:])):
            petz_map = PetzMap(
                key_A=key_A,
                key_B=key_B,
                density_matrix_A=higher_level_density_matrices[key_A],
                density_matrix_B=higher_level_density_matrices[key_B],
                sqrt_method=True,
            )
            petz_mapped_density_matrix = petz_map.get_combined_system()
            new_key = petz_map.get_new_key()
            higher_level_density_matrices[(new_key.n, new_key.level)] = (
                petz_mapped_density_matrix
            )
            updated_keys += [(new_key.n, new_key.level)]

        ell = 2 * ell + 1
        keys = updated_keys

    higher_level_density_matrices.kill_all_except(ell)
    return higher_level_density_matrices, ell


def get_all_density_matrices_at_level_from_sequence_of_density_matrices(
    density_matrix_sequence: Sequence[np.ndarray],
) -> LatticeDict:
    """
    Computes all the density matrices at input level determined from the dimension of
    the input density matrices in two steps: (1) increase the level to 2 * input_level + 1
    using Petz map. (2) reduce to level 'input_level' with partial trace. Note that this is
    necessary in this case since we assume the input density matrices to share no subsystem.
    """
    # get the level
    input_level = get_base_2_dim(density_matrix_sequence[0]) - 1
    (
        high_level_density_matrices,
        level,
    ) = build_higher_level_from_sequence_of_density_matrices(
        density_matrix_sequence, level=2 * input_level
    )

    # compute *all* the density matrices at level `input_level`
    input_level_density_matrices = reduce_level_to(
        high_level_density_matrices, input_level
    )

    return input_level_density_matrices


def get_density_matrices_at_level(
    density_matrix_sequence: Sequence[np.ndarray], level: int
) -> LatticeDict:
    input_level = get_base_2_dim(density_matrix_sequence[0]) - 1
    density_matrices_at_input_level = (
        get_all_density_matrices_at_level_from_sequence_of_density_matrices(
            density_matrix_sequence
        )
    )
    if input_level <= level:
        # increase level
        density_matrices_at_level = increment_level_from_to(
            density_matrices_at_input_level, from_level=input_level, to_level=level
        )
    else:
        # reduce level
        density_matrices_at_level = reduce_level_to(
            density_matrices_at_input_level, to_level=level
        )

    return density_matrices_at_level


def increment_level_from_to(
    density_matrices: LatticeDict, from_level: int, to_level: int
) -> LatticeDict:
    higher_level_density_matrices = density_matrices.deepcopy()
    for ell in range(from_level, to_level):
        higher_level_density_matrices = get_higher_level_single_processing(
            higher_level_density_matrices, ell
        )

    return higher_level_density_matrices


def reduce_level_to(density_matrices: LatticeDict, to_level: int) -> LatticeDict:
    lower_level_density_matrices = density_matrices.deepcopy()
    level = lower_level_density_matrices.get_max_level()
    while level > to_level:
        lower_level_density_matrices = compute_lower_level_sparse(
            lower_level_density_matrices, level
        )
        level -= 1

    return lower_level_density_matrices


def get_non_overlaping_keys(level: int, length: int) -> list[tuple[float, int]]:
    """
    Compute keys of subsystems that *do not* share common subsystems
    at given `level` and `length`
    """
    non_overlapping_n_values = np.arange(
        level / 2, (level + 1) * length - level / 2, level + 1
    )
    return [(n, level) for n in non_overlapping_n_values]
