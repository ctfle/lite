import pytest
from mock import MagicMock

import numpy as np
from local_information.core.utils import *
from local_information.lattice.lattice_dict import LatticeDict

np.random.seed(seed=42)


class TestUtil:
    @pytest.fixture(scope="function")
    def mock_rho_dict(self, request):
        """
        Mock rho_dict with all necessary functionalities.
        """
        density_matrix, n_min, n_max, level = request.param
        mock = MagicMock()
        mock.__getitem__.return_value = density_matrix
        mock.boundaries.return_value = (n_min, n_max)
        n_values = np.arange(n_min, n_max + 1)
        keys = [(j, level) for j in n_values]
        vals = [density_matrix for _ in range(len(n_values))]

        mock.keys.return_value = [(j, level) for j in np.arange(n_min, n_max + 1)]
        mock.deepcopy.return_value = LatticeDict.from_list(keys, vals)
        return mock

    @pytest.mark.parametrize(
        "matrix, level",
        [
            (np.eye(2) / 2, 0),
            (np.eye(4) / 4, 1),
            (np.eye(8) / 8, 2),
            (np.eye(16) / 16, 3),
        ],
    )
    def test_get_higher_level_1(self, matrix, level):
        keys = [(n, level) for n in range(10)]
        vals = [matrix for _ in range(10)]
        test_dict = LatticeDict.from_list(keys, vals)
        higher_level_sqrt = get_higher_level(test_dict, level=level, sqrt_method=True)
        higher_level_exp = get_higher_level(test_dict, level=level, sqrt_method=False)
        # assert higher_level_exp == higher_level_sqrt
        for key, val in higher_level_sqrt.items_at_level(level + 1):
            assert np.allclose(val, np.eye(2 ** (level + 2)) / (2 ** (level + 2)))

    @pytest.mark.parametrize(
        "matrix, level",
        [
            (np.eye(2) / 2, 0),
            (np.eye(4) / 4, 1),
            (np.eye(8) / 8, 2),
            (np.eye(16) / 16, 3),
        ],
    )
    def test_get_higher_level_n_min_equals_n_max(self, matrix, level):
        keys = [(0, level)]
        vals = [matrix]
        test_dict = LatticeDict.from_list(keys, vals)
        higher_level_sqrt = get_higher_level(test_dict, level=level, sqrt_method=True)
        higher_level_exp = get_higher_level(test_dict, level=level, sqrt_method=False)
        assert higher_level_exp == higher_level_sqrt
        assert list(higher_level_exp.keys()) == []

    @pytest.mark.parametrize(
        "level, system_size",
        [
            (0, 10),
            (0, 100),
            (0, 200),
        ],
    )
    def test_get_higher_level_random_matrix(self, level, system_size):
        # Note: using random matrices, this tests does only makes sense at level 0
        # for higher levels the matrices share subsystems and random matrices don't suffice anymore
        keys = [(n, level) for n in range(system_size)]
        vals = []
        for _ in range(system_size):
            matrix = np.diag(np.random.uniform(size=2 ** (level + 1)))
            matrix /= np.trace(matrix)
            vals += [matrix]

        test_dict = LatticeDict.from_list(keys, vals)
        higher_level_sqrt = get_higher_level(test_dict, level=level, sqrt_method=True)
        higher_level_exp = get_higher_level(test_dict, level=level, sqrt_method=False)
        assert higher_level_exp == higher_level_sqrt

        # assert that all partial traces yield the correct matrices
        for n in range(system_size - 1):
            trace_right_exp = ptrace(
                higher_level_exp[(0.5 + n, level + 1)], 1, end="right"
            )
            trace_right_sqrt = ptrace(
                higher_level_sqrt[(0.5 + n, level + 1)], 1, end="right"
            )
            assert np.allclose(trace_right_exp, trace_right_sqrt)
            assert np.allclose(test_dict[(n, level)], trace_right_exp)

    @pytest.mark.parametrize(
        "matrix, level, result",
        [
            (np.array([[0.5, 0.0], [0.0, 0.5]]), 0, np.eye(4) / 4),
            (
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                0,
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ),
            (
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                0,
                np.array(
                    [
                        [0.81, 0.0, 0.0, 0.0],
                        [0.0, 0.09, 0.0, 0.0],
                        [0.0, 0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
            ),
        ],
    )
    def test_get_higher_level_2(self, matrix, level, result):
        keys = [(n, level) for n in range(10)]
        vals = [matrix for _ in range(10)]
        test_dict = LatticeDict.from_list(keys, vals)
        higher_level_sqrt = get_higher_level(test_dict, level=level, sqrt_method=True)

        higher_level_exp = get_higher_level(test_dict, level=level, sqrt_method=True)
        assert higher_level_exp == higher_level_sqrt
        for key, val in higher_level_sqrt.items_at_level(level + 1):
            assert np.allclose(val, result)

    @pytest.mark.parametrize(
        "matrix_1, matrix_2, level, result_1, result_2, result_3",
        [
            (
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                0,
                np.array(
                    [
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                np.array(
                    [
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.5, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ),
            (
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                0,
                np.array(
                    [
                        [0.45, 0.0, 0.0, 0.0],
                        [0.0, 0.45, 0.0, 0.0],
                        [0.0, 0.0, 0.05, 0.0],
                        [0.0, 0.0, 0.0, 0.05],
                    ]
                ),
                np.array(
                    [
                        [0.45, 0.0, 0.0, 0.0],
                        [0.0, 0.05, 0.0, 0.0],
                        [0.0, 0.0, 0.45, 0.0],
                        [0.0, 0.0, 0.0, 0.05],
                    ]
                ),
                np.array(
                    [
                        [0.81, 0.0, 0.0, 0.0],
                        [0.0, 0.09, 0.0, 0.0],
                        [0.0, 0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
            ),
            (
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                0,
                np.array(
                    [
                        [0.45, 0.0, 0.0, 0.0],
                        [0.0, 0.05, 0.0, 0.0],
                        [0.0, 0.0, 0.45, 0.0],
                        [0.0, 0.0, 0.0, 0.05],
                    ]
                ),
                np.array(
                    [
                        [0.45, 0.0, 0.0, 0.0],
                        [0.0, 0.45, 0.0, 0.0],
                        [0.0, 0.0, 0.05, 0.0],
                        [0.0, 0.0, 0.0, 0.05],
                    ]
                ),
                np.array(
                    [
                        [0.25, 0.0, 0.0, 0.0],
                        [0.0, 0.25, 0.0, 0.0],
                        [0.0, 0.0, 0.25, 0.0],
                        [0.0, 0.0, 0.0, 0.25],
                    ]
                ),
            ),
        ],
    )
    def test_get_higher_level_heterogeneous_system(
        self, matrix_1, matrix_2, level, result_1, result_2, result_3
    ):
        keys = [(n, level) for n in range(9)]
        vals = [matrix_1 for _ in range(9)]
        vals[4] = matrix_2
        test_dict = LatticeDict.from_list(keys, vals)
        higher_level_sqrt = get_higher_level(test_dict, level=level, sqrt_method=True)

        higher_level_exp = get_higher_level(test_dict, level=level, sqrt_method=True)

        assert higher_level_exp == higher_level_sqrt
        for key, val in higher_level_sqrt.items_at_level(level + 1):
            if key == (3.5, 1):
                assert np.allclose(val, result_1)
            elif key == (4.5, 1):
                assert np.allclose(val, result_2)
            else:
                assert np.allclose(val, result_3)

    @pytest.mark.parametrize(
        "mock_rho_dict, level, mutual_information",
        [
            ((np.eye(4) / 4, 0, 9, 1), 1, 0.0),
            (
                (
                    np.array(
                        [
                            [0.5, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.5],
                        ]
                    ),
                    0,
                    9,
                    1,
                ),
                1,
                np.log(2),
            ),
        ],
        indirect=["mock_rho_dict"],
    )
    def test_compute_mutual_information_at_level_1(
        self, mock_rho_dict, level, mutual_information
    ):
        return_dict, mut_inf_dict = compute_mutual_information_at_level(
            mock_rho_dict, level
        )
        for key, val in mut_inf_dict.items():
            assert np.allclose(val, mutual_information)

    @pytest.mark.parametrize(
        "mock_rho_dict, level",
        [
            (
                (
                    np.array(
                        [
                            [0.45, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.55],
                        ]
                    ),
                    0,
                    9,
                    1,
                ),
                1,
            ),
            (
                (
                    np.array(
                        [
                            [0.25, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.75],
                        ]
                    ),
                    0,
                    9,
                    1,
                ),
                1,
            ),
            (
                (
                    np.array(
                        [
                            [0.15, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.85],
                        ]
                    ),
                    0,
                    9,
                    1,
                ),
                1,
            ),
            (
                (
                    np.array(
                        [
                            [0.05, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.95],
                        ]
                    ),
                    0,
                    9,
                    1,
                ),
                1,
            ),
            (
                (
                    np.array(
                        [
                            [0.81, 0.0, 0.0, 0.0],
                            [0.0, 0.09, 0.0, 0.0],
                            [0.0, 0.0, 0.09, 0.0],
                            [0.0, 0.0, 0.0, 0.01],
                        ]
                    ),
                    0,
                    9,
                    1,
                ),
                1,
            ),
        ],
        indirect=["mock_rho_dict"],
    )
    def test_compute_mutual_information_at_level_2(self, mock_rho_dict, level):
        _, mut_inf_dict = compute_mutual_information_at_level(mock_rho_dict, level)
        # all have the same value
        values = set(mut_inf_dict.values())
        assert len(values) == 1

    @pytest.mark.parametrize(
        "mock_rho_dict, level",
        [
            (
                (
                    np.eye(2 ** (4 + 1)) / 2 ** (4 + 1),
                    0,
                    9,
                    4,
                ),
                4,
            ),
            (
                (
                    np.eye(2 ** (3 + 1)) / 2 ** (3 + 1),
                    0,
                    9,
                    3,
                ),
                3,
            ),
            (
                (
                    np.eye(2 ** (2 + 1)) / 2 ** (2 + 1),
                    0,
                    9,
                    2,
                ),
                2,
            ),
            (
                (
                    np.eye(2 ** (1 + 1)) / 2 ** (1 + 1),
                    0,
                    9,
                    1,
                ),
                1,
            ),
        ],
        indirect=["mock_rho_dict"],
    )
    def test_compute_mutual_information_at_level_is_zero(self, mock_rho_dict, level):
        # check that mutual information is 0 for infinite temperature states
        _, mut_inf_dict = compute_mutual_information_at_level(mock_rho_dict, level)
        # all values are 0 exactly
        for value in mut_inf_dict.values():
            assert np.allclose(value, 0.0)

    @pytest.mark.parametrize(
        "density_matrix, lower_level, level, n_min, n_max",
        [
            (np.eye(4) / 4, np.array([[0.5, 0.0], [0.0, 0.5]]), 1, 0, 10),
            (
                np.array(
                    [
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.5],
                    ]
                ),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                1,
                0,
                10,
            ),
            (
                np.array(
                    [
                        [0.81, 0.0, 0.0, 0.0],
                        [0.0, 0.09, 0.0, 0.0],
                        [0.0, 0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                1,
                0,
                10,
            ),
        ],
    )
    def test_compute_lower_level(
        self, density_matrix, lower_level, level, n_min, n_max
    ):
        n_values = np.arange(n_min, n_max + 1)
        keys = [(j, level) for j in n_values]
        vals = [density_matrix for _ in range(len(n_values))]
        mock_rho_dict = LatticeDict.from_list(keys, vals)

        return_dict = compute_lower_level(mock_rho_dict, level)

        for key, val in return_dict.items_at_level(level - 1):
            assert len(val) == 2**level
            assert np.allclose(lower_level, val)

    @pytest.mark.parametrize(
        "density_matrix, information, level, n_min, n_max",
        [
            (np.eye(4) / 4, 0, 1, 0, 10),
            (
                np.array(
                    [
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.5],
                    ]
                ),
                np.log(2),
                1,
                0,
                10,
            ),
            (
                np.array(
                    [
                        [0.4, 0.0, 0.0, 0.0],
                        [0.0, 0.1, 0.0, 0.0],
                        [0.0, 0.0, 0.1, 0.0],
                        [0.0, 0.0, 0.0, 0.4],
                    ]
                ),
                0.19274476,
                1,
                0,
                10,
            ),
        ],
    )
    def test_compute_von_Neumann_information(
        self, density_matrix, information, level, n_min, n_max
    ):
        n_values = np.arange(n_min, n_max + 1)
        keys = [(j, level) for j in n_values]
        vals = [density_matrix for _ in range(len(n_values))]
        mock_rho_dict = LatticeDict.from_list(keys, vals)

        information_dict = compute_von_Neumann_information(mock_rho_dict, level)
        for key, val in information_dict.items():
            assert np.allclose(val, information)

    @pytest.mark.parametrize(
        "density_matrix, level, n_min, n_max, push",
        [
            (np.eye(4) / 4, 1, 0, 10, 32),
            (
                np.array(
                    [
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.5],
                    ]
                ),
                1,
                0,
                10,
                1,
            ),
            (
                np.array(
                    [
                        [0.81, 0.0, 0.0, 0.0],
                        [0.0, 0.09, 0.0, 0.0],
                        [0.0, 0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
                1,
                0,
                10,
                2,
            ),
            (
                np.array(
                    [
                        [0.8, 0.0, 0.0, 0.0],
                        [0.0, 0.15, 0.0, 0.0],
                        [0.0, 0.0, 0.04, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
                1,
                0,
                10,
                25,
            ),
        ],
    )
    def test_push_keys(self, density_matrix, level, n_min, n_max, push):
        n_values = np.arange(n_min, n_max + 1)
        keys = [(j, level) for j in n_values]
        vals = [density_matrix for _ in range(len(n_values))]
        mock_rho_dict = LatticeDict.from_list(keys, vals)
        n_min_init, n_max_init = mock_rho_dict.boundaries(level)
        pushed_dict = push_keys(mock_rho_dict, number=push)
        n_min_pushed, n_max_pushed = pushed_dict.boundaries(level)
        assert n_min_pushed - push == n_min_init
        assert n_max_pushed - push == n_max_init

        for key in keys:
            n = key[0]
            assert (n + push, level) in pushed_dict.keys()

    def test_one_shift(self):
        return

    def test_information_gradient(self):
        pass
