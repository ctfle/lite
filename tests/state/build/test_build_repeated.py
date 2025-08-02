import numpy as np
import pytest

from local_information.core.utils import compute_lower_level
from local_information.state.build.build_finite_state import get_non_overlaping_keys
from local_information.state.build.build_repeated_elements import (
    get_boundaries,
    get_repetitions,
)


class TestBuildRepeated:
    @pytest.fixture(scope="function")
    def random_density_matrix(self, request) -> np.ndarray:
        """
        Creates a list of density matrices which are random but all identical.
        """
        level = request.param
        density_matrix = np.random.uniform(size=2 ** (level + 1))
        return np.diag(density_matrix / np.sum(density_matrix))

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

    @pytest.mark.parametrize(
        "random_density_matrix, input_level, build_level",
        [
            (
                (1),
                1,
                5,
            ),
            (
                (2),
                2,
                5,
            ),
            (
                (3),
                3,
                5,
            ),
            (
                (4),
                4,
                5,
            ),
        ],
        indirect=["random_density_matrix"],
    )
    def test_get_repeated_lower_level_density_matrices_for_random_density_matrix(
        self, random_density_matrix, input_level, build_level
    ):
        """
        Tests the construction of the repeated elements using a single random density matrix
        """
        # this is required to get the deduced keys that gets
        # associated to the input sequence of density matrices
        repetitions = get_repetitions(
            density_matrix_sites=input_level + 1, level=build_level
        )
        dedeuced_keys = get_non_overlaping_keys(input_level, repetitions)

        repeated_matrices = get_boundaries([random_density_matrix], max_l=build_level)
        assert repeated_matrices[0].get_max_level() == build_level

        # in this tests the repeated_matrices only contain one element
        lower_level = repeated_matrices[0].deepcopy()
        level = build_level
        while level > input_level:
            lower_level = compute_lower_level(lower_level, level)
            level -= 1

        # check that we get back the input matrix by partial trace
        for i, (key, density_matrix) in enumerate(lower_level.items()):
            if key in dedeuced_keys:
                assert np.allclose(density_matrix, random_density_matrix)

    @pytest.mark.parametrize(
        "random_density_matrix_tuple, input_level, build_level",
        [
            (
                (1),
                1,
                5,
            ),
            (
                (2),
                2,
                5,
            ),
            (
                (3),
                3,
                5,
            ),
            (
                (4),
                4,
                5,
            ),
        ],
        indirect=["random_density_matrix_tuple"],
    )
    def test_get_repeated_lower_level_density_matrices_for_tuple_of_random_density_matrix(
        self, random_density_matrix_tuple, input_level, build_level
    ):
        """
        Tests the construction of the repeated elements using a tuple of random density matrix
        """
        # this is required to get the deduced keys that gets
        # associated to the input sequence of density matrices
        repetitions = get_repetitions(
            density_matrix_sites=input_level + 1, level=build_level
        )
        dedeuced_keys = get_non_overlaping_keys(input_level, repetitions)

        repeated_matrices = get_boundaries(
            random_density_matrix_tuple, max_l=build_level
        )
        assert repeated_matrices[0].get_max_level() == build_level

        # check both density matrix Lattice dicts (describing left and right repeated elements)
        for j in range(2):
            lower_level = repeated_matrices[j].deepcopy()
            level = build_level
            while level > input_level:
                lower_level = compute_lower_level(lower_level, level)
                level -= 1

            # check that we get back the input matrix by partial trace
            for i, (key, density_matrix) in enumerate(lower_level.items()):
                if key in dedeuced_keys:
                    assert np.allclose(density_matrix, random_density_matrix_tuple[j])
