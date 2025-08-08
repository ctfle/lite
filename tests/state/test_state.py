import os
import shutil

import numpy as np
import pytest

from local_information.lattice.lattice_dict import LatticeDict
from local_information.state.state import State


class TestState:
    @pytest.fixture
    def test_state_asymptotic(self):
        test_system = [
            [np.array([[0.2, 0.0], [0.0, 0.8]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        state = State.build(test_system, 1)
        return state

    @pytest.fixture
    def test_state_asymptotic_invalid(self):
        test_system = [
            [np.array([[0.3, 0.0], [0.0, 0.8]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        state = State.build(test_system, 1)
        return state

    @pytest.fixture
    def test_state_inf_temp(self):
        test_system = [
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        state = State.build(test_system, 5)
        return state

    @pytest.fixture
    def test_state_finite(self):
        test_system = [
            [
                np.array([[0.2, 0.0], [0.0, 0.8]]),
                np.array([[0.8, 0.0], [0.0, 0.2]]),
                np.array([[0.2, 0.0], [0.0, 0.8]]),
            ],
            [],
        ]
        state = State.build(test_system, 1)
        return state

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
        "bulk_density_matrix_sequence, boundary_density_matrix_sequence, level, level_overhead",
        [
            (
                [
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.8, 0.0], [0.0, 0.2]]),
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                ],
                [
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
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
                [
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
                    np.array([[0.2, 0.0], [0.0, 0.8]]),
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
                [
                    np.array([[0.4, 0.0], [0.0, 0.6]]),
                    np.array([[0.6, 0.0], [0.0, 0.4]]),
                ],
                0,
                3,
            ),
        ],
    )
    def test_build_asymptotic_level_overhead(
        self,
        bulk_density_matrix_sequence,
        boundary_density_matrix_sequence,
        level,
        level_overhead,
    ):
        state = State.build_asymptotic(
            bulk_density_matrices=bulk_density_matrix_sequence,
            boundary_density_matrices=boundary_density_matrix_sequence,
            level_overhead=level_overhead,
        )
        assert state.dyn_max_l == level + level_overhead
        assert state.density_matrix.get_max_level() == level + level_overhead

    def test_state_extend(self, test_state_finite, test_state_asymptotic):
        """check the state extend"""

        finite_state_dict = test_state_finite.density_matrix
        assert len(finite_state_dict.keys()) == 2
        # check that there are only keys at level 1
        assert finite_state_dict.keys_at_level(1)
        assert finite_state_dict.dim_at_level(0) == 0

        for key, density_matrix in finite_state_dict.items():
            assert np.allclose(np.trace(density_matrix), 1.0)

        asymp_state_dict = test_state_asymptotic.density_matrix
        assert len(asymp_state_dict.keys()) == 4
        # check that there are only keys at level 1
        assert asymp_state_dict.keys_at_level(1)
        assert asymp_state_dict.dim_at_level(0) == 0

        for key, density_matrix in asymp_state_dict.items():
            assert np.allclose(np.trace(density_matrix), 1.0)

    def test_boundaries(self, test_state_asymptotic):
        """check that the boundary terms are sufficient"""
        boundaries = np.eye(4) / 4
        n_min, n_max = test_state_asymptotic.density_matrix.boundaries(1)
        for key, val in test_state_asymptotic.density_matrix.items_at_level(1):
            if key[0] in [n_min, n_max]:
                assert np.allclose(val, boundaries)
        pass

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_asymptotic_state_construction(self, level):
        """check that for each given level"""
        system = [
            [np.array([[0.4, 0.0], [0.0, 0.6]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        state = State.build(system, level)
        # assert the state build up to the right level
        assert len(state.density_matrix.keys()) == 3 + 2 * level - level
        assert state.density_matrix.keys_at_level(level)
        for key, val in state.density_matrix.items():
            assert key[1] == level

        # assert the boundaries have the right form
        boundary_term = np.eye(2 ** (level + 1)) / 2 ** (level + 1)
        n_min, n_max = state.density_matrix.boundaries(level)
        for key, val in state.density_matrix.items_at_level(level):
            if key[0] in [n_min, n_max]:
                assert np.allclose(val, boundary_term)

    def test_inf_temp(self, test_state_inf_temp):
        """assert all terms are infinite temperature"""
        # assert the boundaries have the right form
        boundary_term = np.eye(2 ** (5 + 1)) / 2 ** (5 + 1)
        for key, val in test_state_inf_temp.density_matrix.items_at_level(5):
            assert np.allclose(val, boundary_term)

    @pytest.mark.parametrize(
        "system",
        [
            ([[np.array([[0.3, 0.0], [0.0, 0.8]])], []]),
        ],
    )
    def test_invalid(self, system):
        """invalid state construction is not possible"""
        with pytest.raises(Exception):
            State.build(system, 1)

    def test_empty(self):
        """invalid state construction is not possible"""
        system = [[], []]
        with pytest.raises(Exception):
            State.build(system, 1)

    def test_save_checkpoint_load_checkpoint(
        self, test_state_finite, test_state_asymptotic
    ):
        test_state_finite.starting_time = 0.12345
        folder = "test_folder"
        test_state_finite.save_checkpoint(folder)
        # save checkpoint
        assert os.path.isdir("test_folder")
        assert os.path.isfile("test_folder/state.pkl")
        assert os.path.isfile("test_folder/state_meta_data.yaml")

        # load checkpoint
        loaded_state = State.from_checkpoint(folder)
        assert loaded_state == test_state_finite
        assert loaded_state.anchor == test_state_finite.anchor
        assert loaded_state.starting_time == test_state_finite.starting_time
        shutil.rmtree("test_folder")

        test_state_asymptotic.starting_time = 0.12345
        folder = "test_folder"
        test_state_asymptotic.save_checkpoint(folder)
        # save checkpoint
        assert os.path.isdir("test_folder")
        assert os.path.isfile("test_folder/state.pkl")
        assert os.path.isfile("test_folder/state_meta_data.yaml")

        # load checkpoint
        loaded_state = State.from_checkpoint(folder)
        assert loaded_state == test_state_asymptotic
        assert loaded_state.anchor == test_state_asymptotic.anchor
        assert loaded_state.starting_time == test_state_asymptotic.starting_time
        shutil.rmtree("test_folder")

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
    def test_reduce_level(self, density_matrix, lower_level, level, n_min, n_max):
        n_values = np.arange(n_min, n_max + 1)
        keys = [(j, level) for j in n_values]
        vals = [density_matrix for _ in range(len(n_values))]
        mock_rho_dict = LatticeDict.from_list(keys, vals)
        state = State(density_matrix=mock_rho_dict, case="finite")
        state.reduce_to_level(level=level - 1)

        for key, val in state.density_matrix.items():
            assert np.allclose(val, lower_level)
        assert state.dyn_max_l == level - 1

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
    def test_reduce_level_pop_boundary(
        self, density_matrix, lower_level, level, n_min, n_max
    ):
        n_values = np.arange(n_min, n_max + 1)
        keys = [(j, level) for j in n_values]
        vals = [density_matrix for _ in range(len(n_values))]
        mock_rho_dict = LatticeDict.from_list(keys, vals)
        n_min_init, n_max_init = mock_rho_dict.boundaries(level)
        state = State(density_matrix=mock_rho_dict, case="finite")
        state.reduce_to_level(level=level - 1, pop_boundary=True)

        for key, val in state.density_matrix.items():
            assert np.allclose(val, lower_level)

        assert len(state.density_matrix.keys()) == len(n_values) - 1
        assert state.density_matrix.boundaries(level - 1) == (
            n_min_init + 0.5,
            n_max_init - 0.5,
        )
