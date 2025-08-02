from __future__ import annotations

from typing import Sequence

import numpy as np
from local_information.lattice.lattice_dict import LatticeDict
from local_information.state.build.build_finite_state import (
    get_density_matrices_at_level,
)
from local_information.state.state_helper_funcs import get_base_2_dim


def get_boundaries(
    density_matrix_sequence: Sequence[np.ndarray], max_l: int
) -> list[LatticeDict]:
    """
    This function discriminates the two cases of having
    len(dens_matrix_list)==1 (left and right end coincide)
    and len(dens_matrix_list)==2 (they are distinct).
    Computes the density matrices at max_l.
    """
    if len(density_matrix_sequence) == 1:
        density_matrices_at_level_max_l = compute_repeated_elements(
            density_matrix_sequence[0], max_l
        )
        repeated_elements = [density_matrices_at_level_max_l]

    # asymptotic invariant
    elif len(density_matrix_sequence) == 2:
        repeated_elements_left = compute_repeated_elements(
            density_matrix_sequence[0], max_l, orientation="left"
        )
        repeated_elements_right = compute_repeated_elements(
            density_matrix_sequence[1], max_l, orientation="right"
        )
        repeated_elements = [repeated_elements_left, repeated_elements_right]

    else:
        raise AssertionError(
            "the boundaries must be given as a sequence with 1 or 2 elements"
        )

    return repeated_elements


def compute_repeated_elements(
    density_matrix: np.ndarray, max_l: int, orientation: str = "right"
) -> LatticeDict:
    """
    Computes the required density matrices at level max_l
    """
    # get the lattice size of dens_matrix
    size = get_base_2_dim(density_matrix)
    # get number of lattice sites needed to represent the operators at max_l
    repetitions = get_repetitions(density_matrix_sites=size, level=max_l)

    density_matrix_sequence = [density_matrix] * repetitions
    density_matrices_at_max_l = get_density_matrices_at_level(
        density_matrix_sequence=density_matrix_sequence, level=max_l
    )
    # store only the density matrices required for time evolution:
    # the leftmost size ones for orientation ='right'
    # and the rightmost size ones for orientation ='left'
    store_sites_keys = []
    if orientation == "right":
        for j in range(size):
            store_sites_keys += [(max_l / 2 + j, max_l)]

    elif orientation == "left":
        n_max = max_l + size
        n_0 = n_max - 1 - max_l / 2
        for j in range(size):
            store_sites_keys += [(n_0 - j, max_l)]

    # drop the rest
    density_matrices_at_max_l.kill_all_except(max_l)
    for key in density_matrices_at_max_l.keys_at_level(max_l):
        if key not in store_sites_keys:
            density_matrices_at_max_l.pop(key)

    return density_matrices_at_max_l


def get_repetitions(density_matrix_sites: int, level: int) -> int:
    # get min number of lattice sites needed to represent the operators at max_l
    n_max = level + density_matrix_sites
    repetitions = n_max // density_matrix_sites
    if n_max % density_matrix_sites != 0:
        repetitions += 1

    return repetitions
