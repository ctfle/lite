import pytest
from scipy.linalg import expm

from local_information.core.runge_kutta_solvers.local_runge_kutta import *
from local_information.core.runge_kutta_solvers.runge_kutta_parameters import (
    RK810_parameters,
    RK1012_parameters,
)
from local_information.core.utils import compute_lower_level
from local_information.core.utils import get_higher_level
from local_information.operators.hamiltonian import Hamiltonian
from local_information.operators.lindbladian import Lindbladian
from local_information.state.state import State
from local_information.lattice.lattice_dict import LatticeDict


class TestRungeKutta:
    @pytest.fixture
    def test_trivial_hamiltonian(self):
        J_list = [0 for _ in range(11)]
        hamiltonian_couplings = [["zz", J_list]]
        max_l = 5
        return Hamiltonian(max_l, hamiltonian_couplings)

    @pytest.fixture
    def test_single_particle_hamiltonian(self):
        x_list = [1 for _ in range(11)]
        hamiltonian_couplings = [["x", x_list]]
        max_l = 5
        return Hamiltonian(max_l, hamiltonian_couplings)

    @pytest.fixture
    def test_mnay_particle_hamiltonian_zz(self):
        J_list = [1 for _ in range(5)]
        hamiltonian_couplings = [["zz", J_list]]
        max_l = 5
        return Hamiltonian(max_l, hamiltonian_couplings)

    @pytest.fixture
    def test_mnay_particle_hamiltonian_xx(self):
        J_list = [1 for _ in range(5)]
        hamiltonian_couplings = [["xx", J_list]]
        max_l = 5
        return Hamiltonian(max_l, hamiltonian_couplings)

    @pytest.fixture
    def test_trivial_lindbladian(self):
        J_list = [0 for _ in range(11)]
        hamiltonian_couplings = [["zz", J_list]]
        max_l = 5
        jump_couplings = [["z", J_list]]
        return Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

    @pytest.fixture
    def test_single_particle_lindbladian(self):
        x_list = [1 for _ in range(11)]
        hamiltonian_couplings = [["x", x_list]]
        max_l = 5
        jump_couplings = [["z", x_list]]
        return Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

    @pytest.fixture
    def test_state(self):
        system = [
            [np.array([[0.4, 0.0], [0.0, 0.6]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        return State.build(system, 1)

    @pytest.fixture
    def test_config(self):
        config = RungeKuttaConfig()
        return config

    def test_runge_kutta_trivial(
        self, test_config, test_trivial_hamiltonian, test_state
    ):
        range_ = 1
        time = 0
        test_rk = LocalRungeKuttaSolver(
            runge_kutta_config=test_config,
            range_=range_,
            hamiltonian=test_trivial_hamiltonian,
        )
        new_dens_mat, new_time = test_rk.solve(test_state)

        for key, density_matrix in new_dens_mat.items():
            assert np.allclose(density_matrix, test_state.density_matrix[key])

        assert new_time > time

    def test_runge_kutta_final_time(
        self, test_config, test_trivial_hamiltonian, test_state
    ):
        range_ = 1
        time = 0
        small_time_step = 0.001
        test_rk = LocalRungeKuttaSolver(
            runge_kutta_config=test_config,
            range_=range_,
            hamiltonian=test_trivial_hamiltonian,
        )
        new_dens_mat, new_time = test_rk.solve(test_state, final_time=small_time_step)

        for key, density_matrix in new_dens_mat.items():
            assert np.allclose(density_matrix, test_state.density_matrix[key])

        assert new_time == small_time_step

    def test_runge_kutta_single_body_hamiltonian(
        self, test_config, test_single_particle_hamiltonian, test_state
    ):
        range_ = 1
        time = 0
        test_rk = LocalRungeKuttaSolver(
            runge_kutta_config=test_config,
            range_=range_,
            hamiltonian=test_single_particle_hamiltonian,
        )
        new_dens_mat, new_time = test_rk.solve(test_state)

        # check that no information leaked towards larger scales
        # neither anything spread to initially infinite temperature sites

        assert test_state.density_matrix.keys() == new_dens_mat.keys()
        assert np.allclose(test_state.density_matrix[(0.5, 1)], np.eye(4) / 4)
        assert np.allclose(test_state.density_matrix[(3.5, 1)], np.eye(4) / 4)

    def test_runge_kutta_single_body_lindbladian(
        self, test_config, test_single_particle_lindbladian, test_state
    ):
        range_ = 1
        time = 0
        test_rk = LocalLindbladRungeKuttaSolver(
            runge_kutta_config=test_config,
            range_=range_,
            lindbladian=test_single_particle_lindbladian,
        )
        new_dens_mat, new_time = test_rk.solve(test_state)

        # check that no information leaked towards larger scales
        # neither anything spread to initially infinite temperature sites
        assert test_state.density_matrix.keys() == new_dens_mat.keys()
        assert np.allclose(new_dens_mat[(0.5, 1)], np.eye(4) / 4)
        assert np.allclose(new_dens_mat[(3.5, 1)], np.eye(4) / 4)

    def test_runge_kutta_mnay_body_hamiltonian_xx(
        self, test_config, test_mnay_particle_hamiltonian_xx, test_state
    ):
        range_ = 1
        time = 0
        test_rk = LocalRungeKuttaSolver(
            runge_kutta_config=test_config,
            range_=range_,
            hamiltonian=test_mnay_particle_hamiltonian_xx,
        )
        test_rk.step_size = 0.015
        new_dens_mat, new_time = test_rk.solve(test_state)
        # compare the solution to the exact solution for the given system
        # to get the exact solution we require the matrix exponential of the Hamiltonian

        # full system Hamiltonian
        H_full_system = test_mnay_particle_hamiltonian_xx.subsystem_hamiltonian[
            (2.0, 4)
        ]

        # compute the time evolution operator
        U = expm(-1j * new_time * H_full_system.toarray())

        # compute state
        density_matrices = test_state.density_matrix
        for ell in range(4):
            density_matrices += get_higher_level(density_matrices, 1 + ell)

        # state of the full system has key (2.0, 4)
        full_system_state = density_matrices[(2.0, 4)]

        # compute time evolution
        exact_evolved_state = U @ full_system_state @ np.conjugate(np.transpose(U))

        exact_state = LatticeDict.from_list([(2.0, 4)], [exact_evolved_state])
        # get the subsystems of the exact evolved state
        for ell in range(3):
            exact_state = compute_lower_level(exact_state, 4 - ell)

        for key, val in exact_state.items():
            assert np.max(np.abs(exact_state[key] - new_dens_mat[key])) < 1e-6

    def test_runge_kutta_mnay_body_hamiltonian_zz(
        self, test_config, test_mnay_particle_hamiltonian_zz, test_state
    ):
        range_ = 1
        test_rk = LocalRungeKuttaSolver(
            runge_kutta_config=test_config,
            range_=range_,
            hamiltonian=test_mnay_particle_hamiltonian_zz,
        )
        test_rk.step_size = 0.015
        new_dens_mat, new_time = test_rk.solve(test_state)

        # compare the solution to the exact solution for the given system
        # to get the exact solution we require the matrix exponential of the Hamiltonian

        # full system Hamiltonian
        H_full_system = test_mnay_particle_hamiltonian_zz.subsystem_hamiltonian[
            (2.0, 4)
        ]

        # compute the time evolution operator
        U = expm(-1j * new_time * H_full_system.toarray())

        # compute state
        density_matrices = test_state.density_matrix
        for ell in range(4):
            density_matrices += get_higher_level(density_matrices, 1 + ell)

        # state of the full system has key (2.0, 4)
        full_system_state = density_matrices[(2.0, 4)]

        # compute time evolution
        exact_evolved_state = U @ full_system_state @ np.conjugate(np.transpose(U))

        exact_state = LatticeDict.from_list([(2.0, 4)], [exact_evolved_state])
        # get the subsystems of the exact evolved state
        for ell in range(3):
            exact_state = compute_lower_level(exact_state, 4 - ell)

        for key, val in exact_state.items():
            assert np.max(np.abs(exact_state[key] - new_dens_mat[key])) < 1e-6

    def test_parameter_loading_1012(self):
        a, b, c = RK1012_parameters()
        assert len(a) == 25
        assert len(b) == 25
        assert len(c) == 25
        assert np.linalg.norm(a) != 0
        assert np.linalg.norm(b) != 0
        assert np.linalg.norm(c) != 0

    def test_parameter_loading_810(self):
        a, b, c = RK810_parameters()
        assert np.linalg.norm(a) != 0
        assert np.linalg.norm(b) != 0
        assert np.linalg.norm(c) != 0
