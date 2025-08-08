from __future__ import annotations

from local_information.core.utils import compute_lower_level_sparse, align_to_level
from local_information.lattice.lattice_dict import LatticeDict
from local_information.state.build.build_finite_state import increment_level_from_to
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    pass


def concatenate_bulk_and_boundaries(
    bulk_density_matrices: LatticeDict,
    boundaries: list[LatticeDict],
) -> LatticeDict:
    """
    This function computes the combined LatticeDict by shifted concatenation of
    structure part and repeated parts. We assume that all density matrices are
    available at the same level.
    """
    repeated_left = boundaries[0]
    repeated_right = boundaries[1]

    # get all density matrices on the same level
    bulk_level = bulk_density_matrices.get_max_level()
    repeated_left = align_to_level(repeated_left, bulk_level)
    repeated_right = align_to_level(repeated_right, bulk_level)

    number_of_sites_required_for_repeated_left = len(repeated_left) + bulk_level
    number_of_sites_required_for_structure = len(bulk_density_matrices) + bulk_level

    combined = LatticeDict()

    combined += repeated_left

    shift_to_structure = number_of_sites_required_for_repeated_left
    for key, structure_density_matrix in bulk_density_matrices.items():
        (n, ell) = key
        combined[(n + shift_to_structure, ell)] = structure_density_matrix

    shift_to_repeated_right = (
        number_of_sites_required_for_repeated_left
        + number_of_sites_required_for_structure
    )
    for key, repeated_right_density_matrix in repeated_right.items():
        (n, ell) = key
        combined[(n + shift_to_repeated_right, ell)] = repeated_right_density_matrix

    return combined


def extract_left_and_right_boundaries(boundary_density_matrices: Sequence[LatticeDict]):
    if len(boundary_density_matrices) == 1:
        repeated_left = boundary_density_matrices[0]
        repeated_right = boundary_density_matrices[0]
    elif len(boundary_density_matrices) == 2:
        repeated_left = boundary_density_matrices[0]
        repeated_right = boundary_density_matrices[1]
    else:
        raise ValueError(
            "sequence of boundary density matrices must have 1 or 2 elements."
        )

    return [repeated_left, repeated_right]


def get_combined(
    bulk: LatticeDict, boundaries: list[LatticeDict], level: int
) -> LatticeDict:
    """
    This functions 'fills the gaps' between the bulk and the boundary parts
    to get all the density matrices at the build level (specified by `level`).
    Note: bulk and boundary input do not necessarily have the same level here.
    We construct the lowest level density matrices and
    use Petz map to increase the level back to build level.
    TODO: this can be done more efficiently by building only the missing parts
    (and not everything as it is done here)
    """
    boundaries = extract_left_and_right_boundaries(boundaries)
    bulk_boundary_concatenation = concatenate_bulk_and_boundaries(bulk, boundaries)
    # after concatenation all the level of the density matrices is that of the
    # input bulk density matrices
    bulk_level = bulk_boundary_concatenation.get_max_level()
    while bulk_level > 0:
        bulk_boundary_concatenation = compute_lower_level_sparse(
            bulk_boundary_concatenation, bulk_level
        )
        bulk_level -= 1

    combined = increment_level_from_to(
        bulk_boundary_concatenation, from_level=0, to_level=level
    )
    return combined
