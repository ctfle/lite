import os
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from local_information.config.config import TimeEvolutionConfig
from local_information.lindblad_evolution import OpenSystem
from local_information.operators.lindbladian import Lindbladian
from local_information.state.state import State
from local_information.config.monitor import DataContainer, DataConfig
from local_information.lattice.lattice_dict import LatticeDict


class TestOpenSystem:
    @pytest.fixture(scope="function")
    def test_hamiltonian(self, request):
        max_l = 5
        (L, J, hL, hT) = request.param
        J_list = [J for _ in range(L)]
        hT_list = [hT for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [["zz", J_list], ["z", hL_list], ["x", hT_list]]
        jump_couplings = []
        return Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

    @pytest.fixture(scope="function")
    def test_random_hamiltonian(self, request):
        max_l = 5
        L = 10
        (J_x, J_y, J_z, h_x, h_y, h_z) = request.param
        J_x_list = [J_x for _ in range(L)]
        J_y_list = [J_x for _ in range(L)]
        J_z_list = [J_x for _ in range(L)]
        hx_list = [h_x for _ in range(L)]
        hy_list = [h_y for _ in range(L)]
        hz_list = [h_z for _ in range(L)]
        hamiltonian_couplings = [
            ["zz", J_z_list],
            ["xx", J_x_list],
            ["yy", J_y_list],
            ["x", hx_list],
            ["y", hy_list],
            ["z", hz_list],
        ]
        jump_couplings = []
        return Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

    @pytest.fixture(scope="function")
    def test_random_lindbladian(self, request):
        max_l = 5
        L = 10
        (J_x, J_y, J_z, h_x, h_y, h_z, j_p, j_m, j_z) = request.param
        J_x_list = [J_x for _ in range(L)]
        J_y_list = [J_x for _ in range(L)]
        J_z_list = [J_x for _ in range(L)]
        hx_list = [h_x for _ in range(L)]
        hy_list = [h_y for _ in range(L)]
        hz_list = [h_z for _ in range(L)]
        hamiltonian_couplings = [
            ["zz", J_z_list],
            ["xx", J_x_list],
            ["yy", J_y_list],
            ["x", hx_list],
            ["y", hy_list],
            ["z", hz_list],
        ]
        # lindblad part
        plus_minus_list = [j_p for _ in range(L)]
        jz_list = [j_z for _ in range(L)]
        jump_couplings = [
            ["+", plus_minus_list],
            ["-", plus_minus_list],
            ["z", jz_list],
        ]

        return Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

    @pytest.fixture
    def test_state_1(self):
        # asymptotic state
        system = [
            [np.array([[0.4, 0.0], [0.0, 0.6]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        return State.build(system, 1)

    @pytest.fixture
    def test_state_2(self):
        # state defined for the entire system
        L = 10
        system_matrices = [np.array([[0.4, 0.0], [0.0, 0.6]]) for _ in range(L)]
        system = [system_matrices, []]
        return State.build(system, 1)

    @pytest.fixture
    def test_state_inf_temp(self):
        # asymptotic state
        system = [
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        return State.build(system, 1)

    @pytest.fixture
    def test_data_container_1(self):
        data_config = DataConfig(observables=[some_func, some_other_func])
        return DataContainer(config=data_config)

    @pytest.fixture
    def test_closed_system(self):
        # some initial state
        system = [
            [np.array([[0.4, 0.0], [0.0, 0.6]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        test_state = State.build(system, 1)

        # hamiltonian
        max_l = 5
        L = 10
        J = 0.25
        hL = 0.125
        hT = -0.2625
        J_list = [J for _ in range(L)]
        hT_list = [hT for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [["zz", J_list], ["z", hL_list], ["x", hT_list]]
        jump_couplings = []
        test_hamiltonian = Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

        # data container
        data_config = DataConfig(observables=[some_func, some_other_func])
        data_container = DataContainer(config=data_config)
        close_system = OpenSystem(
            test_state,
            test_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder"
            ),
            data=data_container,
        )

        return close_system

    @pytest.mark.parametrize(
        "test_hamiltonian",
        [
            (5, 1.0, 0.9045, 0.805),
            (6, 1.0, 0.9045, 0.805),
            (7, 1.0, 0.9045, 0.805),
            (10, 1.0, 0.9045, 0.805),
            (11, 2.0, 0.435, 0.75),
            (12, 0.5, 0.0, 0.75),
            (13, 1.0, 0.9345, 0.0),
            (14, 2.0, 1.5, 0.75),
            (15, 0.5, 2.435, 0.765),
            (20, 0.5, 0.435, 2.765),
            (30, 5.5, 0.435, 2.765),
        ],
        indirect=["test_hamiltonian"],
    )
    def test_time_evolution_step_1(
        self, test_hamiltonian, test_state_1, test_data_container_1
    ):
        test_closed_system = OpenSystem(
            init_state=test_state_1,
            lindbladian=test_hamiltonian,
            config=TimeEvolutionConfig(save_checkpoint=True),
            data=test_data_container_1,
        )
        # make sure system is large enough so that sites cn be added
        # initial keys
        initial_keys = deepcopy(list(test_closed_system.state.density_matrix.keys()))

        # choose small evolution time to ensure single step
        test_closed_system._time_evolution_step(final_time=0.001)
        # check the keys in state.density_matrix
        for key in initial_keys:
            # all initial keys must be contained after the time evolution step
            assert key in test_closed_system.state.density_matrix.keys()
        assert len(initial_keys) <= len(test_closed_system.state.density_matrix.keys())

        pass

    @pytest.mark.parametrize(
        "test_hamiltonian, additional_sites, added",
        [
            ((5, 1.0, 0.9345, 0.0), 2, 0),
            ((7, 2.0, 1.5, 0.75), 2, 1),
            ((9, 0.5, 2.435, 0.765), 2, 2),
        ],
        indirect=["test_hamiltonian"],
    )
    def test_enlarge_system_2(
        self,
        test_hamiltonian,
        additional_sites,
        added,
        test_state_1,
        test_data_container_1,
    ):
        # system size is small so not all sites can actually be added
        test_closed_system = OpenSystem(
            init_state=test_state_1,
            lindbladian=test_hamiltonian,
            config=TimeEvolutionConfig(save_checkpoint=True),
            data=test_data_container_1,
        )
        # initial keys
        initial_keys = deepcopy(list(test_closed_system.state.density_matrix.keys()))
        n_min, n_max = test_closed_system.state.density_matrix.boundaries(1)
        # choose small evolution time to ensure single step
        test_closed_system._enlarge_state(additional_sites=additional_sites)
        # check the keys in state.density_matrix
        assert len(test_closed_system.state.density_matrix.keys()) == 2 * added + len(
            initial_keys
        )
        (
            enlarged_n_min,
            enlarged_n_max,
        ) = test_closed_system.state.density_matrix.boundaries(1)
        assert n_min == enlarged_n_min + added
        assert n_max == enlarged_n_max - added
        pass

    @pytest.mark.parametrize(
        "test_hamiltonian, additional_sites",
        [
            ((20, 1.0, 0.9345, 0.0), 1),
            ((20, 2.0, 1.5, 0.75), 2),
            ((20, 0.5, 2.435, 0.765), 3),
            ((20, 0.5, 2.435, 0.765), 4),
            ((5, 0.5, 2.435, 0.765), 4),
            ((7, 0.5, 2.435, 0.765), 4),
            ((9, 0.5, 2.435, 0.765), 4),
        ],
        indirect=["test_hamiltonian"],
    )
    def test_check_convergence(
        self, test_hamiltonian, additional_sites, test_state_1, test_data_container_1
    ):
        # tests the removal of sites
        test_closed_system = OpenSystem(
            init_state=test_state_1,
            lindbladian=test_hamiltonian,
            config=TimeEvolutionConfig(save_checkpoint=True),
            data=test_data_container_1,
        )
        # initial keys
        initial_keys = deepcopy(list(test_closed_system.state.density_matrix.keys()))
        n_min_init, n_max_init = test_closed_system.state.density_matrix.boundaries(1)
        # choose small evolution time to ensure single step
        enlarged_left, enlarged_right = test_closed_system._enlarge_state(
            additional_sites=additional_sites
        )
        tolerance = test_closed_system.config.system_size_tol
        test_closed_system.state.check_convergence(
            sites_to_check_left=enlarged_left,
            sites_to_check_right=enlarged_right,
            tolerance=tolerance,
        )
        # check that all added sites are remove again
        assert len(test_closed_system.state.density_matrix.keys()) == len(initial_keys)
        n_min, n_max = test_closed_system.state.density_matrix.boundaries(1)
        assert n_min == n_min_init
        assert n_max == n_max_init
        pass

    @pytest.mark.parametrize(
        "keys, state_density_matrices, shift_value",
        [
            ([(1, 0), (2.5, 1)], [np.eye(2) / 2, np.eye(4) / 4], 1),
            (
                [(1, 0), (2, 0), (2.5, 1)],
                [
                    np.array([[0.8, 0.0], [0.0, 0.2]]),
                    np.array([[1.0, 0.0], [0.0, 0.0]]),
                    np.array(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                ],
                10,
            ),
            (
                [(1, 0), (2, 0), (2.5, 1)],
                [
                    np.array([[0.8, 0.0], [0.0, 0.2]]),
                    np.array([[1.0, 0.0], [0.0, 0.0]]),
                    np.array(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                ],
                1000,
            ),
        ],
    )
    def test_shift_unshift(
        self, keys, state_density_matrices, shift_value, test_closed_system
    ):
        test_closed_system.config.shift = shift_value
        with patch.object(
            test_closed_system.state,
            "density_matrix",
            LatticeDict.from_list(keys, state_density_matrices),
        ):
            # shift
            shift_dict = test_closed_system._shift()
            for key, val in shift_dict.items():
                assert np.allclose(np.trace(val), shift_value)

            for key, val in test_closed_system.state.density_matrix.items():
                assert np.allclose(np.trace(val), 1)
                for element in np.diag(val):
                    assert element != 0
            # unshift and demand identity
            test_closed_system._unshift(shift_dict)
            assert (
                LatticeDict.from_list(keys, state_density_matrices)
                == test_closed_system.state.density_matrix
            )
        pass

    def test_measure(self, test_closed_system):
        # all functionalities in measure are tested already
        # just tests it works
        values = test_closed_system.measure(return_values=True)
        assert len(values) == 6
        pass

    def test_save_checkpoint_load_checkpoint(self, test_closed_system):
        test_closed_system.measure()
        # save checkpoint
        test_closed_system.save_checkpoint()
        assert os.path.isdir("test_folder")
        assert os.path.isfile("test_folder/data_config.yaml")
        assert os.path.isfile("test_folder/config.yaml")
        assert os.path.isfile("test_folder/state_meta_data.yaml")
        assert os.path.isfile("test_folder/diffusion_const.pkl")
        assert os.path.isfile("test_folder/diffusion_length.pkl")
        assert os.path.isfile("test_folder/times.pkl")
        assert os.path.isfile("test_folder/system_size.pkl")
        assert os.path.isfile("test_folder/some_func.pkl")
        assert os.path.isfile("test_folder/some_other_func.pkl")
        assert os.path.isfile("test_folder/state.pkl")
        assert os.path.isfile("test_folder/lindbladian.pkl")

        # load checkpoint
        working_dir = (
            Path(__file__).parent.resolve().as_posix() + "/test_time_evolution.py"
        )
        loaded_closed_system = OpenSystem.from_checkpoint(
            folder="test_folder", module_path=working_dir
        )
        assert loaded_closed_system.lindbladian == test_closed_system.lindbladian
        assert loaded_closed_system.state == test_closed_system.state

        shutil.rmtree("test_folder")
        pass

    def test_attach_to_existing_file(self, test_closed_system):
        # all functions are already tested separately
        # just tests it works
        test_closed_system.measure()
        # save checkpoint
        test_closed_system.save_checkpoint()
        test_closed_system._time_evolution_step(final_time=0.001)
        test_closed_system.measure()
        test_closed_system.attach_to_existing_file()
        assert 1

        shutil.rmtree("test_folder")
        pass

    def test_evolve_1(self, test_closed_system):
        # tests the final time
        test_closed_system.evolve(max_evolution_time=1.0, final_time=True)
        assert test_closed_system.state.current_time == 1.0
        test_closed_system.evolve(max_evolution_time=1.0, final_time=False)
        assert test_closed_system.state.current_time >= 2.0

        # tests runge kutta step-size update
        starting_time = test_closed_system.state.current_time
        test_closed_system.evolve(max_evolution_time=0.001, final_time=True)
        final_time = test_closed_system.state.current_time
        assert np.allclose(final_time - starting_time, 0.001)
        assert test_closed_system.config.runge_kutta_config.step_size >= 0.001
        shutil.rmtree("test_folder")
        pass

    @pytest.mark.parametrize(
        "test_random_hamiltonian",
        [tuple(np.random.uniform() for _ in range(6)) for _ in range(3)],
        indirect=["test_random_hamiltonian"],
    )
    def test_time_invariant_state_evolution(
        self, test_random_hamiltonian, test_state_inf_temp, test_data_container_1
    ):
        # this tests is actually provided by tests of state
        # assemble ClosedSystem object
        test_closed_system = OpenSystem(
            init_state=test_state_inf_temp,
            lindbladian=test_random_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder"
            ),
            data=test_data_container_1,
        )
        # assert state does not change
        test_closed_system.evolve(max_evolution_time=1.0)
        assert test_closed_system.state == test_state_inf_temp
        shutil.rmtree("test_folder")
        pass

    @pytest.mark.parametrize(
        "test_random_hamiltonian",
        [tuple(np.random.uniform() for _ in range(6)) for _ in range(3)],
        indirect=["test_random_hamiltonian"],
    )
    def test_full_non_trivial_evolution(
        self, test_random_hamiltonian, test_state_1, test_data_container_1
    ):
        # tests evolution of non-trivial state with no trivial hamiltonian
        # assemble ClosedSystem object
        test_closed_system = OpenSystem(
            init_state=test_state_1,
            lindbladian=test_random_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder"
            ),
            data=test_data_container_1,
        )
        test_closed_system.evolve(max_evolution_time=3.0, final_time=True)
        assert np.allclose(test_closed_system.state.current_time, 3.0)
        # continue evolution
        test_closed_system.evolve(max_evolution_time=1.0, final_time=True)
        assert test_closed_system.state.current_time == 4.0
        shutil.rmtree("test_folder")
        pass

    @pytest.mark.parametrize(
        "test_hamiltonian, system_size",
        [
            ((10, 2.0, 1.5045, 1.65), 10),
            ((9, 1.0, 2.5045, 0.65), 9),
            ((10, 0.0, 1.545, 2.65), 10),
        ],
        indirect=["test_hamiltonian"],
    )
    def test_update_lindbladian(
        self, test_hamiltonian, test_closed_system, system_size
    ):
        # update hamiltonian
        try:
            test_closed_system.lindbladian = test_hamiltonian
        except AssertionError:
            assert system_size != 10
        else:
            assert test_closed_system.solver._system_operator == test_hamiltonian

    @pytest.mark.parametrize(
        "test_hamiltonian",
        [
            (5, 2.0, 1.5045, 1.65),
            (10, 1.0, 2.5045, 0.65),
            (11, 1.0, 2.5045, 0.65),
            (20, 0.0, 1.545, 2.65),
            (21, 0.0, 1.545, 2.65),
        ],
        indirect=["test_hamiltonian"],
    )
    def test_align(self, test_state_1, test_hamiltonian, test_data_container_1):
        # tests the correct alignment of state and Hamiltonian
        # the state should be set in the middle of area where
        # the Hamiltonian is defined.

        # assemble ClosedSystem object
        test_closed_system = OpenSystem(
            init_state=test_state_1,
            lindbladian=test_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder"
            ),
            data=test_data_container_1,
        )
        # anchor is middle of the system
        if test_hamiltonian.L % 2 == 0:
            assert test_hamiltonian.L / 2 - 1 == test_closed_system.state.anchor
        else:
            assert (test_hamiltonian.L - 1) / 2 == test_closed_system.state.anchor

        # check the keys of state
        # assert all keys to be in the region where the Hamiltonian is defined
        for key in test_closed_system.state.density_matrix.keys():
            assert key in test_hamiltonian.subsystem_hamiltonian.keys()

        pass

    @pytest.mark.parametrize(
        "test_hamiltonian",
        [
            (10, 2.0, 1.5045, 1.65),
            (10, 1.0, 1.045, 2.65),
            (11, 1.0, 1.045, 2.65),
            (12, 1.0, 1.045, 2.65),
        ],
        indirect=["test_hamiltonian"],
    )
    def test_align_finite(self, test_state_2, test_hamiltonian, test_data_container_1):
        # tests the correct alignment of state and Hamiltonian
        # for the state defined on the full system

        if test_hamiltonian.L != 10:
            # system and hamiltonian are now not
            # defined on the same region. Should throw and error
            with pytest.raises(ValueError):
                OpenSystem(
                    init_state=test_state_2,
                    lindbladian=test_hamiltonian,
                    config=TimeEvolutionConfig(
                        save_checkpoint=True, checkpoint_folder="test_folder"
                    ),
                    data=test_data_container_1,
                )
        else:
            # assemble ClosedSystem object
            test_closed_system = OpenSystem(
                init_state=test_state_2,
                lindbladian=test_hamiltonian,
                config=TimeEvolutionConfig(
                    save_checkpoint=True, checkpoint_folder="test_folder"
                ),
                data=test_data_container_1,
            )
            # n0 is middle of the system
            # since now the state has to be defined on the entire
            # region of the Hamiltonian which fixes n0
            # for L = 10 we have n0 = 4.5
            assert test_hamiltonian.L / 2 - 0.5 == test_closed_system.state.anchor

            # check the keys of state
            # assert all keys to be in the region where the Hamiltonian is defined
            for key in test_closed_system.state.density_matrix.keys():
                assert key in test_hamiltonian.subsystem_hamiltonian.keys()

    def test_minimize(self, test_closed_system):
        pass

    @pytest.mark.parametrize(
        "test_random_lindbladian",
        [tuple(np.random.uniform() for _ in range(9)) for _ in range(3)],
        indirect=["test_random_lindbladian"],
    )
    def test_lindbladian_evolution(
        self, test_random_lindbladian, test_state_1, test_data_container_1
    ):
        """tests evolution of non-trivial state with no trivial hamiltonian"""
        # assemble ClosedSystem object
        test_open_system = OpenSystem(
            init_state=test_state_1,
            lindbladian=test_random_lindbladian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder"
            ),
            data=test_data_container_1,
        )
        # change some parameters
        test_open_system.evolve(max_evolution_time=1.0, final_time=True)
        assert np.allclose(test_open_system.state.current_time, 1.0)
        # continue evolution
        test_open_system.evolve(max_evolution_time=1.0, final_time=True)
        assert test_open_system.state.current_time == 2.0
        shutil.rmtree("test_folder")
        pass

    @pytest.mark.parametrize(
        "test_random_lindbladian",
        [tuple(np.random.uniform() for _ in range(9)) for _ in range(3)],
        indirect=["test_random_lindbladian"],
    )
    def test_lindbladian_evolution(
        self, test_random_lindbladian, test_state_1, test_data_container_1
    ):
        # tests evolution of non-trivial state with no trivial hamiltonian
        # assemble ClosedSystem object
        test_open_system = OpenSystem(
            init_state=test_state_1,
            lindbladian=test_random_lindbladian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder"
            ),
            data=test_data_container_1,
        )
        # change some parameters
        test_open_system.solver.config.max_error = 1e-10
        test_open_system.solver.config.RK_order = "23"
        test_open_system.evolve(max_evolution_time=1.0, final_time=True)
        assert np.allclose(test_open_system.state.current_time, 1.0)
        # continue evolution
        test_open_system.evolve(max_evolution_time=1.0, final_time=True)
        assert test_open_system.state.current_time == 2.0
        shutil.rmtree("test_folder")
        pass


def some_func(rho):
    return 1


def some_other_func(rho):
    return 1
