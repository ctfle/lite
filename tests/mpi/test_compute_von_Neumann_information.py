from tests.mpi.utils import *
from local_information.state.state import State
from local_information.core.utils import (
    compute_von_Neumann_information_single_processing,
    compute_von_Neumann_information,
    get_higher_level_single_processing,
)
from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()

np.random.seed(42)

# to tests run `mpirun -n 2 python -m pytest --with-mpi test_compute_von_Neumann_information.py`


class TestGetHigherLevel:
    @staticmethod
    def update_level(state: State, number_of_updates: int):
        level = state.dyn_max_l
        higher_level_single_processing = state.density_matrix
        for update in range(number_of_updates):
            higher_level_single_processing = get_higher_level_single_processing(
                input_dict=higher_level_single_processing, level=level + update
            )
        return higher_level_single_processing

    @staticmethod
    def compute_von_Neumann_information_and_compare(
        density_matrices: LatticeDict, level
    ):
        von_Neumann_information_single_processing = (
            compute_von_Neumann_information_single_processing(
                rho_dict=density_matrices, level=level
            )
        )

        von_Neumann_information_mpi = compute_von_Neumann_information(
            input_dict=density_matrices, level=level
        )

        if RANK == 0:
            for key, info in von_Neumann_information_mpi.items_at_level(level):
                assert key in von_Neumann_information_single_processing
                assert np.allclose(info, von_Neumann_information_single_processing[key])

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("number_of_updates", [1, 2, 3])
    def test_compute_von_Neumann_information_with_mpi_1(
        self, test_state_1, number_of_updates
    ):
        """Test that get_higher_level yields the same results using single- and multi-processing."""
        initial_level = test_state_1.dyn_max_l
        higher_level_density_matrices = self.update_level(
            test_state_1, number_of_updates
        )
        self.compute_von_Neumann_information_and_compare(
            higher_level_density_matrices, initial_level + number_of_updates
        )

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("number_of_updates", [1, 2, 3])
    def test_compute_von_Neumann_information_with_mpi__2(
        self, test_state_2, number_of_updates
    ):
        initial_level = test_state_2.dyn_max_l
        higher_level_density_matrices = self.update_level(
            test_state_2, number_of_updates
        )
        self.compute_von_Neumann_information_and_compare(
            higher_level_density_matrices, initial_level + number_of_updates
        )

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("number_of_updates", [1, 2, 3])
    def test_compute_von_Neumann_information_with_mpi_3(
        self, test_state_inf_temp, number_of_updates
    ):
        initial_level = test_state_inf_temp.dyn_max_l
        higher_level_density_matrices = self.update_level(
            test_state_inf_temp, number_of_updates
        )
        self.compute_von_Neumann_information_and_compare(
            higher_level_density_matrices, initial_level + number_of_updates
        )

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("number_of_updates", [1, 2, 3])
    @pytest.mark.parametrize(
        "test_thermal",
        [tuple(np.random.uniform() for _ in range(6))],
        indirect=["test_thermal"],
    )
    def test_compute_von_Neumann_information_with_mpi_4(
        self, test_thermal, number_of_updates
    ):
        initial_level = test_thermal.dyn_max_l
        higher_level_density_matrices = self.update_level(
            test_thermal, number_of_updates
        )
        self.compute_von_Neumann_information_and_compare(
            higher_level_density_matrices, initial_level + number_of_updates
        )
