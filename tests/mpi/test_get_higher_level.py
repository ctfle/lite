import numpy as np
import pytest
from tests.mpi.utils import *
from local_information.state.state import State
from local_information.core.utils import (
    get_higher_level,
    get_higher_level_single_processing,
)
from local_information.mpi.mpi_funcs import get_mpi_variables

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()

np.random.seed(42)

# to tests run `mpirun -n 2 python -m pytest --with-mpi test_get_higher_level.py`


class TestGetHigherLevel:
    @staticmethod
    def update_and_compare_results(state: State, number_of_updates: int):
        level = state.dyn_max_l
        higher_level_single_processing = state.density_matrix
        for update in range(number_of_updates):
            higher_level_using_mpi = get_higher_level(
                input_dict=higher_level_single_processing, level=level + update
            )

            higher_level_single_processing = get_higher_level_single_processing(
                input_dict=higher_level_single_processing, level=level + update
            )
            if RANK == 0:
                for key, density_matrix in higher_level_using_mpi.items_at_level(
                    level + update + 1
                ):
                    assert key in higher_level_single_processing
                    assert np.allclose(
                        density_matrix, higher_level_single_processing[key]
                    )

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("number_of_updates", [1, 2, 3])
    def test_get_higher_level_with_mpi_1(self, test_state_1, number_of_updates):
        """Test that get_higher_level yields the same results using single- and multi-processing."""
        self.update_and_compare_results(test_state_1, number_of_updates)

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("number_of_updates", [1, 2, 3])
    def test_get_higher_level_with_mpi_2(self, test_state_2, number_of_updates):
        self.update_and_compare_results(test_state_2, number_of_updates)

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("number_of_updates", [1, 2, 3])
    def test_get_higher_level_with_mpi_3(self, test_state_inf_temp, number_of_updates):
        self.update_and_compare_results(test_state_inf_temp, number_of_updates)

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("number_of_updates", [1, 2, 3])
    @pytest.mark.parametrize(
        "test_thermal",
        [tuple(np.random.uniform() for _ in range(6))],
        indirect=["test_thermal"],
    )
    def test_get_higher_level_with_mpi_4(self, test_thermal, number_of_updates):
        self.update_and_compare_results(test_thermal, number_of_updates)
