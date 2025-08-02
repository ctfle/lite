import os
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch, PropertyMock

import numpy as np
import pytest

from local_information.config import TimeEvolutionConfig
from local_information.operators.hamiltonian import Hamiltonian
from local_information.state.state import State
from local_information.time_evolution import ClosedSystem
from local_information.config.monitor import DataContainer, DataConfig
from local_information.core.petz_map import ptrace
from local_information.lattice.lattice_dict import LatticeDict
from local_information.core.utils import Status

np.random.seed(42)


class TestClosedSystem:
    @pytest.fixture(scope="function")
    def test_hamiltonian(self, request):
        max_l = 5
        (L, J, hL, hT) = request.param
        J_list = [J for _ in range(L)]
        hT_list = [hT for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [["zz", J_list], ["z", hL_list], ["x", hT_list]]
        return Hamiltonian(max_l, hamiltonian_couplings)

    @pytest.fixture(scope="function")
    def test_onsite_hamiltonian(self, request):
        max_l = 3
        (L, hL, hT) = request.param
        hT_list = [hT for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [["z", hL_list], ["x", hT_list]]
        return Hamiltonian(max_l, hamiltonian_couplings)

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
        return Hamiltonian(max_l, hamiltonian_couplings)

    @pytest.fixture(scope="function")
    def test_large_random_hamiltonian(self, request):
        max_l = 5
        L = 100
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
        return Hamiltonian(max_l, hamiltonian_couplings)

    @pytest.fixture
    def test_state_1(self):
        # asymptotic state
        system = [
            [np.array([[0.2, 0.0], [0.0, 0.8]])],
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

    @pytest.fixture(scope="function")
    def test_thermal(self, request):
        max_l = 5
        L = 12
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
        # build the Hamiltonian
        hamiltonian = Hamiltonian(max_l, hamiltonian_couplings)
        # build the state using Hamiltonian
        site_0 = np.eye(2) / 2
        site_1 = (
            np.eye(2**3, dtype=np.complex128)
            - 0.1 * hamiltonian.subsystem_hamiltonian[(L // 2, 2)].toarray()
        ) / 8
        system = [[site_1], [site_0]]
        state = State.build(system, 1)
        return state, hamiltonian

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
        test_hamiltonian = Hamiltonian(max_l, hamiltonian_couplings)

        # data container
        data_config = DataConfig(observables=[some_func, some_other_func])
        data_container = DataContainer(config=data_config)
        close_system = ClosedSystem(
            test_state,
            test_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder"
            ),
            data=data_container,
        )

        return close_system

    @pytest.fixture(scope="function")
    def test_state_update_level(self, request):
        difference, level_overhead = request.param
        # tests state with information controlled by input parameter
        system = [
            [np.array([[0.5 - difference, 0.0], [0.0, 0.5 + difference]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        test_state = State.build(system, level_overhead)
        test_state.loaded_state = True

        return test_state

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
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_hamiltonian,
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
        "test_hamiltonian, additional_sites",
        [
            ((20, 1.0, 0.9045, 0.805), 1),
            ((20, 2.0, 0.435, 0.75), 1),
            ((20, 0.5, 0.0, 0.75), 2),
            ((20, 1.0, 0.9345, 0.0), 2),
            ((20, 2.0, 1.5, 0.75), 2),
            ((20, 0.5, 2.435, 0.765), 3),
            ((20, 0.5, 0.435, 2.765), 3),
            ((20, 5.5, 0.435, 2.765), 4),
        ],
        indirect=["test_hamiltonian"],
    )
    def test_enlarge_system_1(
        self, test_hamiltonian, additional_sites, test_state_1, test_data_container_1
    ):
        # ensure system size is large enough for this tests
        # so that sites can be added at each side
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_hamiltonian,
            config=TimeEvolutionConfig(save_checkpoint=True),
            data=test_data_container_1,
        )
        # initial keys
        initial_keys = deepcopy(list(test_closed_system.state.density_matrix.keys()))
        n_min, n_max = test_closed_system.state.density_matrix.boundaries(1)
        # choose small evolution time to ensure single step
        test_closed_system._enlarge_state(additional_sites=additional_sites)
        # check the keys in state.density_matrix
        assert len(
            test_closed_system.state.density_matrix.keys()
        ) == 2 * additional_sites + len(initial_keys)
        (
            enlarged_n_min,
            enlarged_n_max,
        ) = test_closed_system.state.density_matrix.boundaries(1)
        assert n_min == enlarged_n_min + additional_sites
        assert n_max == enlarged_n_max - additional_sites
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
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_hamiltonian,
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
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_hamiltonian,
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

    def test_measure(self, test_closed_system):
        # all functionalities in measure are tested already
        # just tests it works
        values = test_closed_system.measure(return_values=True)
        assert len(values) == 6

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
        assert os.path.isfile("test_folder/hamiltonian.pkl")

        # load checkpoint
        working_dir = (
            Path(__file__).parent.resolve().as_posix() + "/test_time_evolution.py"
        )
        loaded_closed_system = ClosedSystem.from_checkpoint(
            folder="test_folder", module_path=working_dir
        )
        assert loaded_closed_system.hamiltonian == test_closed_system.hamiltonian
        assert loaded_closed_system.state == test_closed_system.state

        shutil.rmtree("test_folder")

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
        test_closed_system = ClosedSystem(
            init_state=test_state_inf_temp,
            hamiltonian=test_random_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder"
            ),
            data=test_data_container_1,
        )
        # assert state does not change
        test_closed_system.evolve(max_evolution_time=1.0)
        assert test_closed_system.state == test_state_inf_temp
        shutil.rmtree("test_folder")

    @pytest.mark.parametrize(
        "test_random_hamiltonian",
        [tuple(np.random.uniform() for _ in range(6)) for _ in range(3)],
        indirect=["test_random_hamiltonian"],
    )
    def test_full_non_trivial_evolution_1(
        self, test_random_hamiltonian, test_state_1, test_data_container_1
    ):
        # tests evolution of non-trivial state with a non-trivial hamiltonian
        # assemble ClosedSystem object
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_random_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder", shift=1000
            ),
            data=test_data_container_1,
        )
        test_closed_system.solver.config.max_error = 1e-10
        test_closed_system.solver.config.RK_order = "23"
        test_closed_system.evolve(max_evolution_time=3.0, final_time=True)
        assert np.allclose(test_closed_system.state.current_time, 3.0)
        # continue evolution
        test_closed_system.evolve(max_evolution_time=1.0, final_time=True)
        assert test_closed_system.state.current_time == 4.0
        shutil.rmtree("test_folder")

    @pytest.mark.parametrize(
        "test_large_random_hamiltonian",
        [tuple(np.random.uniform() for _ in range(6)) for _ in range(3)],
        indirect=["test_large_random_hamiltonian"],
    )
    def test_full_non_trivial_evolution_check_boundaries(
        self, test_large_random_hamiltonian, test_state_1, test_data_container_1
    ):
        # tests evolution of non-trivial state with a non-trivial hamiltonian
        # check that the boundaries (on the lowest level) are always made up by identities
        # assemble ClosedSystem object
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_large_random_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder", shift=1000
            ),
            data=test_data_container_1,
        )
        for steps in range(6):
            test_closed_system.evolve(max_evolution_time=0.5, final_time=True)
            density_matrix = test_closed_system.state.density_matrix
            # get boundary density matrices
            ell = test_closed_system.state.dyn_max_l
            n_min, n_max = density_matrix.boundaries(ell)
            left = density_matrix[(n_min, ell)]
            right = density_matrix[(n_max, ell)]
            left_end_lowest_level = ptrace(left, spins_to_trace_out=ell, end="right")
            right_end_lowest_level = ptrace(right, spins_to_trace_out=ell, end="left")

            assert np.allclose(left_end_lowest_level, 0.5 * np.eye(2))
            assert np.allclose(right_end_lowest_level, 0.5 * np.eye(2))

    @pytest.mark.parametrize(
        "test_large_random_hamiltonian",
        [tuple(np.random.uniform() for _ in range(6)) for _ in range(3)],
        indirect=["test_large_random_hamiltonian"],
    )
    def test_full_non_trivial_evolution_check_boundaries_using_state(
        self, test_large_random_hamiltonian, test_state_1, test_data_container_1
    ):
        # tests evolution of non-trivial state with a non-trivial hamiltonian
        # check that the boundaries (on the lowest level) are always made up by identities
        # assemble ClosedSystem object
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_large_random_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder", shift=1000
            ),
            data=test_data_container_1,
        )
        for steps in range(6):
            test_closed_system.evolve(max_evolution_time=0.5, final_time=True)
            density_matrix = test_closed_system.state.density_matrix
            # get boundary density matrices
            ell = test_closed_system.state.dyn_max_l
            n_min, n_max = density_matrix.boundaries(ell)
            left = density_matrix[(n_min, ell)]
            right = density_matrix[(n_max, ell)]
            left_end_lowest_level = ptrace(left, spins_to_trace_out=ell, end="right")
            right_end_lowest_level = ptrace(right, spins_to_trace_out=ell, end="left")

            assert np.allclose(left_end_lowest_level, 0.5 * np.eye(2))
            assert np.allclose(right_end_lowest_level, 0.5 * np.eye(2))

    @pytest.mark.parametrize(
        "test_thermal",
        [tuple(np.random.uniform() for _ in range(6))],
        indirect=["test_thermal"],
    )
    def test_thermal_state_random_hamiltonian(
        self, test_thermal, test_data_container_1
    ):
        # assemble ClosedSystem object
        test_closed_system = ClosedSystem(
            init_state=test_thermal[0],
            hamiltonian=test_thermal[1],
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder", shift=100
            ),
            data=test_data_container_1,
        )

        test_closed_system.evolve(max_evolution_time=1.0, final_time=True)
        assert np.allclose(test_closed_system.state.current_time, 1.0)
        # continue evolution
        test_closed_system.evolve(max_evolution_time=1.0, final_time=True)
        assert test_closed_system.state.current_time == 2.0
        shutil.rmtree("test_folder")

    @pytest.mark.parametrize(
        "test_random_hamiltonian",
        [tuple(np.random.uniform() for _ in range(6)) for _ in range(3)],
        indirect=["test_random_hamiltonian"],
    )
    def test_full_non_trivial_evolution_2(
        self, test_random_hamiltonian, test_state_1, test_data_container_1
    ):
        # tests evolution of non-trivial state with no trivial hamiltonian
        # assemble ClosedSystem object
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_random_hamiltonian,
            config=TimeEvolutionConfig(
                save_checkpoint=True, checkpoint_folder="test_folder", shift=10
            ),
            data=test_data_container_1,
        )
        test_closed_system.solver.config.max_error = 1e-8
        test_closed_system.solver.config.RK_order = "45"
        test_closed_system.evolve(max_evolution_time=2.0, final_time=True)
        assert np.allclose(test_closed_system.state.current_time, 2.0)
        # continue evolution
        test_closed_system.evolve(max_evolution_time=1.0, final_time=True)
        assert test_closed_system.state.current_time == 3.0
        shutil.rmtree("test_folder")

    @pytest.mark.parametrize(
        "test_hamiltonian",
        [(10, 2.0, 1.5045, 1.65), (10, 1.0, 2.5045, 0.65), (10, 0.0, 1.545, 2.65)],
        indirect=["test_hamiltonian"],
    )
    def test_update_hamiltonian(self, test_hamiltonian, test_closed_system):
        # update hamiltonian
        test_closed_system.hamiltonian = test_hamiltonian
        assert test_closed_system.solver._system_operator == test_hamiltonian

    @pytest.mark.parametrize(
        "test_hamiltonian",
        [(11, 1.0, 2.5045, 0.65), (12, 0.0, 1.545, 2.65)],
        indirect=["test_hamiltonian"],
    )
    def test_update_hamiltonian_length_diff(self, test_hamiltonian, test_closed_system):
        # update hamiltonian
        with pytest.raises(ValueError):
            test_closed_system.hamiltonian = test_hamiltonian

    @pytest.mark.parametrize(
        "test_hamiltonian",
        [(10, 2.0, 1.5045, 1.65), (10, 1.0, 2.5045, 0.65)],
        indirect=["test_hamiltonian"],
    )
    def test_update_hamiltonian_same_length(self, test_hamiltonian, test_closed_system):
        # update hamiltonian
        test_closed_system.hamiltonian = test_hamiltonian
        assert test_hamiltonian.L == 10

    @pytest.mark.parametrize(
        "test_onsite_hamiltonian",
        [(10, 1.5045, 1.65), (10, 2.5045, 0.65), (10, 1.545, 2.65)],
        indirect=["test_onsite_hamiltonian"],
    )
    def test_max_l_range_when_updating_hamiltonian(
        self, test_onsite_hamiltonian, test_closed_system
    ):
        # update hamiltonian
        test_closed_system.hamiltonian = test_onsite_hamiltonian
        assert test_closed_system.solver._system_operator == test_onsite_hamiltonian
        assert test_closed_system.max_l == test_onsite_hamiltonian.max_l
        assert test_closed_system.range_ == test_onsite_hamiltonian.range_

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
        test_closed_system = ClosedSystem(
            init_state=test_state_1,
            hamiltonian=test_hamiltonian,
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
                ClosedSystem(
                    init_state=test_state_2,
                    hamiltonian=test_hamiltonian,
                    config=TimeEvolutionConfig(
                        save_checkpoint=True, checkpoint_folder="test_folder"
                    ),
                    data=test_data_container_1,
                )
        else:
            # assemble ClosedSystem object
            test_closed_system = ClosedSystem(
                init_state=test_state_2,
                hamiltonian=test_hamiltonian,
                config=TimeEvolutionConfig(
                    save_checkpoint=True, checkpoint_folder="test_folder"
                ),
                data=test_data_container_1,
            )
            # anchor is middle of the system
            # since now the state has to be defined on the entire
            # region of the Hamiltonian which fixes anchnr
            # for L = 10 we have n0 = 4.5
            assert test_hamiltonian.L / 2 - 0.5 == test_closed_system.state.anchor

            # check the keys of state
            # assert all keys to be in the region where the Hamiltonian is defined
            for key in test_closed_system.state.density_matrix.keys():
                assert key in test_hamiltonian.subsystem_hamiltonian.keys()

    @pytest.mark.parametrize(
        "test_state_update_level, threshold, update",
        [
            ((0.1, 0), 0.0001, True),
            ((0.1, 0), 0.001, True),
            ((0.1, 0), 0.01, True),
            ((0.1, 0), 0.1, False),
            ((0.05, 0), 0.0001, True),
            ((0.05, 0), 0.001, True),
            ((0.05, 0), 0.01, False),
            ((0.001, 0), 0.1, False),
            ((0.001, 0), 0.0001, False),
            ((0.001, 0), 0.001, False),
            ((0.001, 0), 0.01, False),
            ((0.001, 0), 0.1, False),
            ((0.1, 1), 0.0001, True),
            ((0.1, 1), 0.001, True),
            ((0.1, 1), 0.01, True),
            ((0.1, 1), 0.1, True),
        ],
        indirect=["test_state_update_level"],
    )
    def test_update_level(self, test_state_update_level, threshold, update):
        """
        Here we tests isolated updates. Triggered if the information passes the threshold.
        Terminates with stop = True when updated.
        """
        with patch(
            "local_information.time_evolution.ClosedSystem.max_l",
            new_callable=PropertyMock,
        ) as mock_max_l:
            mock_max_l.return_value = 1
            with patch(
                "local_information.time_evolution.ClosedSystem.range_",
                new_callable=PropertyMock,
            ) as mock_range_:
                mock_range_.return_value = 1

                J_list = [1 for _ in range(10)]
                hamiltonian_couplings = [
                    ["z", J_list],
                ]
                test_hamiltonian = Hamiltonian(1, hamiltonian_couplings)

                # data container
                data_config = DataConfig(observables=[some_func, some_other_func])
                data_container = DataContainer(config=data_config)
                closed_system = ClosedSystem(
                    test_state_update_level,
                    test_hamiltonian,
                    config=TimeEvolutionConfig(
                        save_checkpoint=False, update_dyn_max_l_threshold=threshold
                    ),
                    data=data_container,
                )

                closed_system._update_dyn_max_l()
                closed_system._set_trigger()
                if update:
                    assert closed_system._trigger.status is Status.Stop
                else:
                    assert closed_system._trigger.status is Status.Continue

    @pytest.mark.parametrize(
        "test_state_update_level, threshold, update",
        [
            ((0.1, 0), 0.0001, True),
            ((0.1, 0), 0.001, True),
            ((0.1, 0), 0.01, True),
            ((0.1, 0), 0.1, False),
            ((0.05, 0), 0.0001, True),
            ((0.05, 0), 0.001, True),
            ((0.05, 0), 0.01, False),
            ((0.001, 0), 0.1, False),
            ((0.001, 0), 0.0001, False),
            ((0.001, 0), 0.001, False),
            ((0.001, 0), 0.01, False),
            ((0.001, 0), 0.1, False),
        ],
        indirect=["test_state_update_level"],
    )
    def test_update_level_check_size_and_number_of_matrices_and(
        self, test_state_update_level, threshold, update
    ):
        """
        Here we tests isolated updates. Triggered if the information passes the threshold.
        Terminates with stop = True when updated.
        """
        with patch(
            "local_information.time_evolution.ClosedSystem.max_l",
            new_callable=PropertyMock,
        ) as mock_max_l:
            mock_max_l.return_value = 1
            with patch(
                "local_information.time_evolution.ClosedSystem.range_",
                new_callable=PropertyMock,
            ) as mock_range_:
                mock_range_.return_value = 1

                J_list = [1 for _ in range(10)]
                hamiltonian_couplings = [
                    ["z", J_list],
                ]
                test_hamiltonian = Hamiltonian(1, hamiltonian_couplings)

                # data container
                data_config = DataConfig(observables=[some_func, some_other_func])
                data_container = DataContainer(config=data_config)
                closed_system = ClosedSystem(
                    test_state_update_level,
                    test_hamiltonian,
                    config=TimeEvolutionConfig(
                        save_checkpoint=False, update_dyn_max_l_threshold=threshold
                    ),
                    data=data_container,
                )
                number_of_elements = len(closed_system.state.density_matrix)

                closed_system._update_dyn_max_l()
                closed_system._set_trigger()
                if closed_system._trigger.status is Status.Stop:
                    assert (
                        len(closed_system.state.density_matrix)
                        == number_of_elements - 1
                    )
                    for matrix in closed_system.state.density_matrix.values_at_level(
                        closed_system.state.dyn_max_l
                    ):
                        assert len(matrix) == 2 ** (closed_system.state.dyn_max_l + 1)

    @pytest.mark.parametrize(
        "test_state_update_level, threshold, max_l",
        [
            ((0.01, 0), 1e-5, 3),
            ((0.05, 0), 1e-5, 4),
            ((0.1, 0), 1e-6, 5),
            ((0.2, 0), 1e-8, 6),
        ],
        indirect=["test_state_update_level"],
    )
    def test_update_only_possible_once(self, test_state_update_level, threshold, max_l):
        """
        After an update, an immediate next update is impossible.
        """
        with patch(
            "local_information.time_evolution.ClosedSystem.max_l",
            new_callable=PropertyMock,
        ) as mock_max_l:
            mock_max_l.return_value = max_l
            with patch(
                "local_information.time_evolution.ClosedSystem.range_",
                new_callable=PropertyMock,
            ) as mock_range_:
                mock_range_.return_value = 1

                J_list = [1 for _ in range(10)]
                hamiltonian_couplings = [
                    ["z", J_list],
                ]
                test_hamiltonian = Hamiltonian(1, hamiltonian_couplings)

                # data container
                data_config = DataConfig(observables=[some_func, some_other_func])
                data_container = DataContainer(config=data_config)
                closed_system = ClosedSystem(
                    test_state_update_level,
                    test_hamiltonian,
                    config=TimeEvolutionConfig(
                        save_checkpoint=False, update_dyn_max_l_threshold=threshold
                    ),
                    data=data_container,
                )
                initial_dyn_max_l = closed_system.state.dyn_max_l

                for _ in range(2):
                    closed_system._update_dyn_max_l()
                    closed_system._set_trigger()
                    assert closed_system._trigger.status is Status.Continue

                assert closed_system.state.dyn_max_l == initial_dyn_max_l + 1


def some_func(rho):
    return 1


def some_other_func(rho):
    return 1
