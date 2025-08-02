import pytest

from local_information.state.build.build_finite_state import *
from local_information.state.build.build_repeated_elements import get_boundaries
from local_information.state.build.build_finite_state import get_finite_state
from local_information.state.build.bulk_boundary_concatenation import (
    get_combined,
    concatenate_bulk_and_boundaries,
)


class TestBuildBulkBoundaryConcatenation:
    @pytest.fixture(scope="function")
    def random_density_matrix_tuple(self, request) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates a list of density matrices which are random but all identical.
        """
        level = request.param
        matrix_left = np.random.uniform(size=2 ** (level + 1))
        matrix_right = np.random.uniform(size=2 ** (level + 1))
        dm_tuple = (
            np.diag(matrix_left / np.sum(matrix_left)),
            np.diag(matrix_right / np.sum(matrix_right)),
        )
        return dm_tuple

    @pytest.fixture(scope="function")
    def sequence_of_different_random_density_matrices(
        self, request
    ) -> list[np.ndarray]:
        """
        Creates a sequence of different random density matrices.
        """
        length, level = request.param
        density_matrix_sequence = [
            np.random.uniform(size=2 ** (level + 1)) for _ in range(length)
        ]
        density_matrix_sequence = [
            np.diag(density_matrix / np.sum(density_matrix))
            for density_matrix in density_matrix_sequence
        ]
        return density_matrix_sequence

    @pytest.mark.parametrize(
        "sequence_of_different_random_density_matrices, random_density_matrix_tuple, input_level, build_level",
        [
            (
                (10, 1),
                (1),
                1,
                5,
            ),
            (
                (10, 2),
                (2),
                2,
                5,
            ),
            (
                (10, 3),
                (3),
                3,
                5,
            ),
            (
                (10, 4),
                (4),
                4,
                5,
            ),
        ],
        indirect=[
            "sequence_of_different_random_density_matrices",
            "random_density_matrix_tuple",
        ],
    )
    def test_bulk_boundary_concatenation(
        self,
        sequence_of_different_random_density_matrices,
        random_density_matrix_tuple,
        input_level,
        build_level,
    ):
        """
        This tests that get_combined is working properly: testing that the combined dict reproduces all
        the lower level density matrices it was build from. The individual parts are tested already.
        So here we tests that the system nicely integrates: We want the density matrices of boundary terms
        and structure to not overlap (i.e. share subsystems) down to the lowest level.
        """
        state_dict, state_level = get_finite_state(
            sequence_of_different_random_density_matrices, max_l=build_level
        )

        repeated_matrices = get_boundaries(
            random_density_matrix_tuple, max_l=build_level
        )
        combined = concatenate_bulk_and_boundaries(
            bulk_density_matrices=state_dict,
            boundaries=repeated_matrices,
        )
        lower_level = combined.deepcopy()
        level = build_level
        while level > 0:
            lower_level = compute_lower_level_sparse(lower_level, level)
            level -= 1

        # tests the system is contiguous
        n_values = lower_level.n_at_level(0)
        for j in range(len(n_values)):
            assert j in n_values

        # assert it has the correct length
        repeated_left_sites = len(repeated_matrices[0]) + build_level
        repeated_right_sites = len(repeated_matrices[1]) + build_level
        structure_sites = len(state_dict) + build_level
        number_of_sites = repeated_right_sites + repeated_left_sites + structure_sites
        assert len(n_values) == number_of_sites

    @pytest.mark.parametrize(
        "sequence_of_different_random_density_matrices, random_density_matrix_tuple, input_level, build_level",
        [
            (
                (10, 1),
                (1),
                1,
                5,
            ),
            (
                (10, 2),
                (2),
                2,
                5,
            ),
            (
                (10, 3),
                (3),
                3,
                5,
            ),
            (
                (10, 4),
                (4),
                4,
                5,
            ),
        ],
        indirect=[
            "sequence_of_different_random_density_matrices",
            "random_density_matrix_tuple",
        ],
    )
    def test_get_combined(
        self,
        sequence_of_different_random_density_matrices,
        random_density_matrix_tuple,
        input_level,
        build_level,
    ):
        """
        Similar to 'test_bulk_boundary_concatenation' but testing additionally that
        we have all density matrices at the build level.
        """
        state_dict, state_level = get_finite_state(
            sequence_of_different_random_density_matrices, max_l=build_level
        )

        repeated_matrices = get_boundaries(
            random_density_matrix_tuple, max_l=build_level
        )
        combined = concatenate_bulk_and_boundaries(
            bulk_density_matrices=state_dict,
            boundaries=repeated_matrices,
        )

        # assert we have all the build level density matrices
        keys = list(combined.keys_at_level(build_level))
        n_min, n_max = combined.boundaries(build_level)
        for n in np.arange(n_min, n_max + 1):
            assert (n, build_level) in keys

        level = build_level
        while level > 0:
            combined = compute_lower_level_sparse(combined, level)
            level -= 1

        # tests the system is contiguous
        n_values = combined.n_at_level(0)
        for j in range(len(n_values)):
            assert j in n_values

        # assert it has the correct length
        repeated_left_sites = len(repeated_matrices[0]) + build_level
        repeated_right_sites = len(repeated_matrices[1]) + build_level
        structure_sites = len(state_dict) + build_level
        number_of_sites = repeated_right_sites + repeated_left_sites + structure_sites
        assert len(n_values) == number_of_sites
