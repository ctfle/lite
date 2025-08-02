import pytest
from mock import MagicMock
from scipy.stats import unitary_group

from local_information.core.minimization.minimization import *
from local_information.core.utils import information_gradient
from local_information.operators.hamiltonian import Hamiltonian
from local_information.lattice.lattice_dict import LatticeDict
from local_information.config.config import TimeEvolutionConfig
from local_information.state.state import State
from local_information.typedefs import SystemOperator


class TestMinimization:
    @staticmethod
    def get_minimizer(hamiltonian: SystemOperator, level: int):
        # we set max_l in TimeEvolutionConfig just to ensure a different value from min_l
        return InformationMinimizer(
            system_operator=hamiltonian,
            config=TimeEvolutionConfig(min_l=level, max_l=level + 2),
        )

    @pytest.fixture(scope="function")
    def mock_rho_dict(self, request):
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

    @pytest.fixture(scope="function")
    def random_hamiltonian(self, request):
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
    def random_lattice_dict(self, request):
        (level, number_of_matrices) = request.param
        density_matrices = []
        keys = []
        for n in range(number_of_matrices):
            random_density_matrix = np.diag(
                np.random.uniform(low=0.01, high=1.0, size=2 ** (level + 1))
            )
            random_density_matrix /= np.trace(random_density_matrix)
            random_unitary = unitary_group.rvs(2 ** (level + 1))
            random_density_matrix = (
                random_unitary.conj().T @ random_density_matrix @ random_unitary
            )
            density_matrices.append(random_density_matrix)
            keys.append((n + 0.5 * level, level))

        return LatticeDict.from_list(keys, density_matrices)

    @pytest.mark.parametrize(
        "mock_rho_dict, level, number_of_matrices, mutual_information",
        [
            ((np.eye(4) / 4, 0, 9, 1), 1, 10, 0.0),
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
                10,
                np.log(2),
            ),
        ],
        indirect=["mock_rho_dict"],
    )
    def test_total_mutual_information(
        self, mock_rho_dict, level, number_of_matrices, mutual_information
    ):
        total_mut_info = total_mutual_information_at_level(level, mock_rho_dict)
        assert total_mut_info == number_of_matrices * mutual_information
        pass

    @pytest.mark.parametrize(
        "random_hamiltonian, density_matrix, level, number_of_matrices",
        [
            (tuple(np.random.uniform() for _ in range(6)), np.eye(16) / 16, 3, 7),
            (tuple(np.random.uniform() for _ in range(6)), np.eye(32) / 32, 4, 6),
            (tuple(np.random.uniform() for _ in range(6)), np.eye(64) / 64, 5, 5),
        ],
        indirect=["random_hamiltonian"],
    )
    def test_minimization_identity(
        self, random_hamiltonian, density_matrix, level, number_of_matrices
    ):
        """tests that minimization yields the correct result"""
        range_ = 1
        dens_mat = [density_matrix for _ in range(number_of_matrices)]
        keys = [(n + level * 0.5, level) for n in range(number_of_matrices)]
        density_matrix_dict = LatticeDict.from_list(keys, dens_mat)
        state = State(density_matrix_dict)
        minimizer = self.get_minimizer(hamiltonian=random_hamiltonian, level=level)
        optimized_density_matrix = minimizer(state, checks=True)
        for ell in range(level - 3, level + 1):
            assert optimized_density_matrix.keys_at_level(ell) is not None
        optimized_density_matrix.kill_all_except(level)
        assert optimized_density_matrix == density_matrix_dict

    @pytest.mark.parametrize(
        "random_hamiltonian, random_lattice_dict, level",
        [
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (4, 6), 4),
            (tuple(np.random.uniform() for _ in range(6)), (5, 5), 5),
        ],
        indirect=["random_hamiltonian", "random_lattice_dict"],
    )
    def test_information_minimizer_information_is_reduced_for_random_density_matrix(
        self, random_hamiltonian, random_lattice_dict, level
    ):
        """
        Test that the InformationMinimizer works as expected:
        lowering information, keeping currents fixed,
        doesn't change lower level density matrices
        """
        _, mut_info_init = compute_mutual_information_at_level(
            random_lattice_dict, level
        )

        state = State(density_matrix=random_lattice_dict)

        information_minimizer = self.get_minimizer(
            hamiltonian=random_hamiltonian, level=level
        )
        rho_dict_after_minimization = information_minimizer(state=state, checks=True)

        for ell in range(level - 3, level + 1):
            assert rho_dict_after_minimization.keys_at_level(ell) is not None
        rho_dict_after_minimization.kill_all_except(level)

        # check mutual information is lowered on input level
        _, mut_info = compute_mutual_information_at_level(
            rho_dict_after_minimization, level
        )
        for key, info in mut_info.items_at_level(level):
            assert info < mut_info_init[key]

    @pytest.mark.parametrize(
        "random_hamiltonian, random_lattice_dict, level",
        [
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (4, 6), 4),
            (tuple(np.random.uniform() for _ in range(6)), (5, 5), 5),
        ],
        indirect=["random_hamiltonian", "random_lattice_dict"],
    )
    def test_information_minimizer_lower_levels_unchanged_for_random_density_matrix(
        self, random_hamiltonian, random_lattice_dict, level
    ):
        """
        Test that the InformationMinimizer works as expected:
        lowering information, keeping currents fixed,
        doesn't change lower level density matrices
        """
        _, mut_info_init = compute_mutual_information_at_level(
            random_lattice_dict, level
        )
        lower_level_dict_init = compute_lower_level(random_lattice_dict, level)

        state = State(density_matrix=random_lattice_dict)

        information_minimizer = self.get_minimizer(
            hamiltonian=random_hamiltonian, level=level
        )
        rho_dict_after_minimization = information_minimizer(state=state, checks=True)
        rho_dict_after_minimization.kill_all_except(level)
        # check the lower level density matrices remain unchanged
        lower_level_dict = compute_lower_level(rho_dict_after_minimization, level)
        assert lower_level_dict == lower_level_dict_init

    @pytest.mark.parametrize(
        "random_hamiltonian, random_lattice_dict, level",
        [
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (4, 6), 4),
            (tuple(np.random.uniform() for _ in range(6)), (5, 5), 5),
        ],
        indirect=["random_hamiltonian", "random_lattice_dict"],
    )
    def test_information_minimizer_information_gradient_unchanged_for_random_density_matrix(
        self, random_hamiltonian, random_lattice_dict, level
    ):
        """
        Test that the InformationMinimizer works as expected:
        lowering information, keeping currents fixed,
        doesn't change lower level density matrices
        """
        range_ = 1
        _, mut_info_init = compute_mutual_information_at_level(
            random_lattice_dict, level
        )
        r = random_lattice_dict.deepcopy()
        r += build_three_lower_levels(r, level)
        n_min, n_max = r.boundaries(level - 1)
        info_current_init = LatticeDict()
        info_current_init += information_gradient(
            r, level - 1, n_min, n_max, range_, random_hamiltonian.subsystem_hamiltonian
        )
        state = State(density_matrix=random_lattice_dict)
        information_minimizer = self.get_minimizer(
            hamiltonian=random_hamiltonian, level=level
        )

        rho_dict_after_minimization = information_minimizer(state=state, checks=True)
        rho_dict_after_minimization.kill_all_except(level)

        # check information gradient remains unchanged
        r = random_lattice_dict.deepcopy()
        r += build_three_lower_levels(r, level)
        info_gradient = information_gradient(
            r, level - 1, n_min, n_max, range_, random_hamiltonian.subsystem_hamiltonian
        )
        for key, currents in info_gradient.items_at_level(level - 1):
            init_currents = info_current_init[key]
            assert np.allclose(init_currents, currents)

    @pytest.mark.parametrize(
        "random_hamiltonian, random_lattice_dict, level",
        [
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (4, 6), 4),
            (tuple(np.random.uniform() for _ in range(6)), (5, 5), 5),
        ],
        indirect=["random_hamiltonian", "random_lattice_dict"],
    )
    def test_information_minimizer_information_current_unchanged_for_random_density_matrix(
        self, random_hamiltonian, random_lattice_dict, level
    ):
        """
        Test that the InformationMinimizer works as expected:
        lowering information, keeping currents fixed,
        doesn't change lower level density matrices
        """
        _, mut_info_init = compute_mutual_information_at_level(
            random_lattice_dict, level
        )

        state = State(density_matrix=random_lattice_dict)
        initial_information_current = state.get_information_current(random_hamiltonian)

        information_minimizer = self.get_minimizer(
            hamiltonian=random_hamiltonian, level=level
        )

        rho_dict_after_minimization = information_minimizer(state=state, checks=True)
        rho_dict_after_minimization.kill_all_except(level)

        # check information current remain unchanged
        state_after_minimization = State(rho_dict_after_minimization)
        information_current_after_minimization = (
            state_after_minimization.get_information_current(random_hamiltonian)
        )
        for (
            key,
            (current_l, current_r),
        ) in information_current_after_minimization.items():
            init_current_l, init_current_r = initial_information_current[key]
            assert np.allclose(current_l, init_current_l)
            assert np.allclose(current_r, init_current_r)


def build_three_lower_levels(rho_dict, ell) -> LatticeDict:
    """
    Computes the two lower levels for the subsequent minimization with keeping the current fix
    """
    lower_level = rho_dict.deepcopy()
    for l in range(3):
        density_matrices_on_lower_level = compute_lower_level(lower_level, ell - l)
        lower_level += density_matrices_on_lower_level
    return lower_level
