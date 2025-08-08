import pytest
import numpy as np
from tests.mpi.utils import *

# to tests run `mpirun -n 2 python -m pytest --with-mpi test_conjugate_gradient_fixed_subsystems.py`


class TestConjugateGradientWithMPI:
    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize(
        "random_hamiltonian, random_lattice_dict, level",
        [
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (4, 6), 4),
            (tuple(np.random.uniform() for _ in range(6)), (5, 5), 5),
        ],
        indirect=["random_hamiltonian", "random_lattice_dict"],
    )
    def test_information_minimizer_lower_levels_unchanged_with_mpi(
        self, random_hamiltonian, random_lattice_dict, level
    ):
        """
        Test that the InformationMinimizer works as expected:
        lowering information, keeping currents fixed,
        doesn't change lower level density matrices
        """
        lower_level_dict_init = compute_lower_level(random_lattice_dict, level)

        state = State(density_matrix=random_lattice_dict)

        information_minimizer = get_minimizer(
            hamiltonian=random_hamiltonian, level=level
        )
        rho_dict_after_minimization = information_minimizer(state=state, checks=True)
        if RANK == 0:
            rho_dict_after_minimization.kill_all_except(level)
            # check the lower level density matrices remain unchanged
            lower_level_dict = compute_lower_level(rho_dict_after_minimization, level)
            assert lower_level_dict == lower_level_dict_init

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize(
        "random_hamiltonian, random_lattice_dict, level",
        [
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
        ],
        indirect=["random_hamiltonian", "random_lattice_dict"],
    )
    def test_information_is_reduced_with_mpi(
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

        information_minimizer = get_minimizer(
            hamiltonian=random_hamiltonian, level=level
        )
        rho_dict_after_minimization = information_minimizer(state=state, checks=True)
        if RANK == 0:
            for ell in range(level - 3, level + 1):
                assert rho_dict_after_minimization.keys_at_level(ell) is not None
            rho_dict_after_minimization.kill_all_except(level)

            # check mutual information is lowered on input level
            _, mut_info = compute_mutual_information_at_level(
                rho_dict_after_minimization, level
            )
            for key, info in mut_info.items_at_level(level):
                assert info < mut_info_init[key]
