import pytest
import numpy as np
from tests.mpi.utils import *

# to tests run `mpirun -n 2 python -m pytest --with-mpi test_conjugate_gradient_fixed_current.py`


class TestConjugateGradientWithMPI:
    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize(
        "random_hamiltonian, random_lattice_dict, level",
        [
            (tuple(np.random.uniform() for _ in range(6)), (3, 7), 3),
            (tuple(np.random.uniform() for _ in range(6)), (4, 6), 4),
            (tuple(np.random.uniform() for _ in range(6)), (5, 5), 5),
            (tuple(np.random.uniform() for _ in range(6)), (5, 7), 5),
        ],
        indirect=["random_hamiltonian", "random_lattice_dict"],
    )
    def test_information_minimizer_information_current_unchanged_for_random_density_matrix(
        self, random_hamiltonian, random_lattice_dict, level
    ):
        """
        Test that the InformationMinimizer works as expected:
        lowering information, keeping currents fixed.
        This tests uses a fake state which is inconsistent: tracing out spins density matrices
        on neighboring density matrices does not result in the same sub-subsystem density matrix.
        This is not a problem when testing single processing. However, with MPI a problem appears
        since the lower levels (on the left  boundary) are build different vs on the right.
        For example with 2 MPI workers and 10 sites we would split the workload where each worker
        takes 5 sites: RANK 0: 0 1 2 3 4, RANK 1: 5 6 7 8 9.
        Now the sub-system density matrix when tracing out the rightmost spin on 4 is not the same as
        compared to tracing out the leftmost site on 5. This is why we exclude these sites in this tests.
        In the actual evolved state this is not a problem (since the state is consistent).
        """
        split_keys = get_split_up_keys(random_lattice_dict, level)
        # get the first element of each list starting from the first list
        first_key_of_each_RANK = list(map(lambda x: x[0], split_keys[1:]))
        keys_to_exclude = list(
            map(lambda x: (x[0] - 0.5, x[1] - 1), first_key_of_each_RANK)
        )
        _, mut_info_init = compute_mutual_information_at_level(
            random_lattice_dict, level
        )

        state = State(density_matrix=random_lattice_dict)
        initial_information_current = state.get_information_current(random_hamiltonian)
        state.density_matrix.kill_all_except(level)
        information_minimizer = get_minimizer(
            hamiltonian=random_hamiltonian, level=level
        )

        rho_dict_after_minimization = information_minimizer(state=state, checks=True)
        rho_dict_after_minimization.kill_all_except(level)

        # check information current remain unchanged
        state_after_minimization = State(rho_dict_after_minimization)
        information_current_after_minimization = (
            state_after_minimization.get_information_current(random_hamiltonian)
        )
        if RANK == 0:
            for (
                key,
                (current_l, current_r),
            ) in information_current_after_minimization.items():
                init_current_l, init_current_r = initial_information_current[key]
                assert np.allclose(current_l, init_current_l)
                if key not in keys_to_exclude:
                    assert np.allclose(current_r, init_current_r)
