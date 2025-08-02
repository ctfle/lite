import pytest

from local_information.state.build.build_finite_state import *
from local_information.state.state import State


class TestBuildFinite:
    @pytest.fixture(scope="function")
    def random_density_matrix_sequence(self, request) -> list[np.ndarray]:
        """
        Creates a list of density matrices which are random but all identical.
        """
        length, level = request.param
        matrix = np.random.uniform(size=2 ** (level + 1))
        density_matrix_sequence = [matrix for _ in range(length)]
        density_matrix_sequence = [
            np.diag(density_matrix / np.sum(density_matrix))
            for density_matrix in density_matrix_sequence
        ]
        return density_matrix_sequence

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
        "density_matrix_sequence, level, level_overhead",
        [
            (
                [
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.8, 0.0], [0.0, 0.2]]),
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                ],
                0,
                1,
            ),
            (
                [
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.4, 0.0], [0.0, 0.6]]),
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.1, 0.0], [0.0, 0.9]]),
                    np.array([[0.3, 0.0], [0.0, 0.7]]),
                ],
                0,
                2,
            ),
            (
                [
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.4, 0.0], [0.0, 0.6]]),
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.1, 0.0], [0.0, 0.9]]),
                    np.array([[0.3, 0.0], [0.0, 0.7]]),
                ],
                0,
                3,
            ),
        ],
    )
    def test_build_finite_level_overhead(
        self, density_matrix_sequence, level, level_overhead
    ):
        state = State.build_finite(
            density_matrices=density_matrix_sequence,
            level_overhead=level_overhead,
        )
        assert state.dyn_max_l == level + level_overhead
        assert state.density_matrix.get_max_level() == level + level_overhead

    @pytest.mark.parametrize(
        "density_matrix_sequence, level, level_overhead",
        [
            (
                [
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.4, 0.0], [0.0, 0.6]]),
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.1, 0.0], [0.0, 0.9]]),
                    np.array([[0.3, 0.0], [0.0, 0.7]]),
                ],
                0,
                5,
            ),
        ],
    )
    def test_finite_level_overhead_to_big_triggers_error(
        self, density_matrix_sequence, level, level_overhead
    ):
        with pytest.raises(ValueError):
            _ = State.build_finite(
                density_matrices=density_matrix_sequence,
                level_overhead=level_overhead,
            )

    @pytest.mark.parametrize(
        "input_density_matrix, length, input_level, build_level",
        [
            (
                np.array(
                    [
                        [0.2, 0.0, 0.0, 0.0],
                        [0.0, 0.1, 0.0, 0.0],
                        [0.0, 0.0, 0.4, 0.0],
                        [0.0, 0.0, 0.0, 0.3],
                    ]
                ),
                10,
                1,
                5,
            )
        ],
    )
    def test_get_finite_state_correct_lower_level_density_matrices(
        self, input_density_matrix, length, input_level, build_level
    ):
        dedeuced_keys = get_non_overlaping_keys(input_level, length)
        density_matrix_sequence = [input_density_matrix for _ in range(length)]
        state_dict, state_level = get_finite_state(
            density_matrix_sequence, max_l=build_level
        )
        assert state_level == build_level
        assert state_dict.get_max_level() == build_level
        lower_level = state_dict.deepcopy()
        level = build_level
        while level > input_level:
            lower_level = compute_lower_level(lower_level, level)
            level -= 1

        for key, density_matrix in lower_level.items():
            if key in dedeuced_keys:
                assert np.allclose(density_matrix, input_density_matrix)

    @pytest.mark.parametrize(
        "random_density_matrix_sequence, input_level, build_level",
        [
            (
                (10, 1),
                1,
                5,
            ),
            (
                (10, 2),
                2,
                5,
            ),
            (
                (10, 3),
                3,
                5,
            ),
            (
                (10, 4),
                4,
                5,
            ),
        ],
        indirect=["random_density_matrix_sequence"],
    )
    def test_get_finite_state_correct_lower_level_density_matrices_for_random_density_matrix(
        self, random_density_matrix_sequence, input_level, build_level
    ):
        """
        All the density matrices in the
        initial sequence of density matrices are equal in this tests
        """
        length = len(random_density_matrix_sequence)
        dedeuced_keys = get_non_overlaping_keys(input_level, length)
        state_dict, state_level = get_finite_state(
            random_density_matrix_sequence, max_l=build_level
        )
        assert state_level == build_level
        assert state_dict.get_max_level() == build_level
        lower_level = state_dict.deepcopy()
        level = build_level
        while level > input_level:
            lower_level = compute_lower_level(lower_level, level)
            level -= 1

        for i, (key, density_matrix) in enumerate(lower_level.items()):
            if key in dedeuced_keys:
                assert np.allclose(density_matrix, random_density_matrix_sequence[0])

    @pytest.mark.parametrize(
        "sequence_of_different_random_density_matrices, input_level, build_level",
        [
            (
                (10, 1),
                1,
                5,
            ),
            (
                (10, 2),
                2,
                5,
            ),
            (
                (10, 3),
                3,
                5,
            ),
            (
                (10, 4),
                4,
                5,
            ),
        ],
        indirect=["sequence_of_different_random_density_matrices"],
    )
    def test_get_finite_state_correct_lower_level_density_matrices_for_sequence_of_random_density_matrix(
        self, sequence_of_different_random_density_matrices, input_level, build_level
    ):
        """
        All the density matrices in the
        initial sequence of density matrices are equal in this tests
        """
        length = len(sequence_of_different_random_density_matrices)
        dedeuced_keys = get_non_overlaping_keys(input_level, length)
        state_dict, state_level = get_finite_state(
            sequence_of_different_random_density_matrices, max_l=build_level
        )
        assert state_level == build_level
        assert state_dict.get_max_level() == build_level
        lower_level = state_dict.deepcopy()
        level = build_level
        while level > input_level:
            lower_level = compute_lower_level(lower_level, level)
            level -= 1

        i = 0
        for key, density_matrix in lower_level.items():
            if key in dedeuced_keys:
                assert np.allclose(
                    density_matrix, sequence_of_different_random_density_matrices[i]
                )
                i += 1
