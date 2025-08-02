"""
To run the tests in this file:
mpirun -n 2 python -m pytest --with-mpi test_distribute_mpi.py
"""

import numpy as np
import pytest

from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.distribute import Distributor
from local_information.mpi.mpi_funcs import get_mpi_variables, print_mpi

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()

np.random.seed(42)


class TestScatterGather:
    @pytest.fixture(scope="function")
    def random_lattice(self, request):
        size, level = request.param
        random_matrix_dict = {
            (n, level): np.random.uniform(size=(2 ** (level + 1), 2 ** (level + 1)))
            for n in range(size)
        }
        return LatticeDict.from_dict(random_matrix_dict)

    @pytest.fixture(scope="function")
    def multi_level_random_lattice(self, request):
        size, level = request.param
        random_matrix_dict = {
            (n, level): np.random.uniform(size=(2 ** (level + 1), 2 ** (level + 1)))
            for n in range(size)
        }
        lattice_at_level = LatticeDict.from_dict(random_matrix_dict)

        random_matrix_dict_higher_level = {
            (n + 0.5, level + 1): np.random.uniform(
                size=(2 ** (level + 1), 2 ** (level + 1))
            )
            for n in range(size - 1)
        }
        lattice_at_higher_level = LatticeDict.from_dict(random_matrix_dict_higher_level)
        return lattice_at_level + lattice_at_higher_level

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize(
        "random_lattice, level",
        [
            ((3, 0), 0),
            ((4, 0), 0),
            ((4, 10), 10),
            ((50, 5), 5),
            ((100, 10), 10),
            ((200, 10), 10),
        ],
        indirect=["random_lattice"],
    )
    def test_scatter(self, random_lattice, level):
        distributor = Distributor(random_lattice, level)
        lattice_on_worker = distributor.scatter(shift=1, number_of_workers=SIZE)
        assert lattice_on_worker
        assert len(list(lattice_on_worker.keys())) > 0

        if RANK == 0:
            print_mpi(RANK, random_lattice)
        print_mpi(RANK, lattice_on_worker)
        # check that each worker has a lattice dict with keys and values of the original LatticeDict
        for key, value in lattice_on_worker.items():
            print_mpi(RANK, lattice_on_worker)
            assert np.allclose(value, random_lattice[key])

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize(
        "random_lattice, level",
        [
            ((3, 0), 0),
            ((4, 0), 0),
            ((4, 10), 10),
            ((50, 5), 5),
            ((100, 10), 10),
            ((200, 10), 10),
        ],
        indirect=["random_lattice"],
    )
    def test_gather(self, random_lattice, level):
        distributor = Distributor(random_lattice, level)
        lattice_on_worker = distributor.scatter(shift=1, number_of_workers=SIZE)
        assert lattice_on_worker
        assert len(list(lattice_on_worker.keys())) > 0

        # check that each worker has a lattice dict with keys and values of the original LatticeDict
        for key, value in lattice_on_worker.items():
            assert np.allclose(value, random_lattice[key])

        # gather and check the root dict is identical to the original
        gathered_lattice = distributor.gather(lattice_on_worker)

        if RANK == 0:
            assert gathered_lattice.keys() == random_lattice.keys()
            for key, value in gathered_lattice.items():
                assert np.allclose(value, random_lattice[key])
        else:
            assert gathered_lattice is None

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize(
        "random_lattice, level",
        [((3, 0), 0), ((4, 0), 0), ((4, 10), 10), ((50, 5), 5)],
        indirect=["random_lattice"],
    )
    def test_distribute_non_root_is_none(self, random_lattice, level):
        distributor = Distributor(random_lattice, level)
        distribution = distributor.distribute(shift=1, number_of_workers=SIZE)
        if RANK != 0:
            assert distribution is None

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize(
        "multi_level_random_lattice, level",
        [((3, 0), 0), ((4, 0), 0), ((4, 5), 5), ((50, 5), 5)],
        indirect=["multi_level_random_lattice"],
    )
    def test_distribute_only_specified_level(self, multi_level_random_lattice, level):
        distributor = Distributor(multi_level_random_lattice, level)
        lattice_on_worker = distributor.scatter(shift=1, number_of_workers=SIZE)
        assert lattice_on_worker
        assert len(list(lattice_on_worker.keys_at_level(level))) > 0
        # check that each worker has a lattice dict with keys and values of the original LatticeDict
        # where each key is associated with the specified level
        for key, value in lattice_on_worker.items():
            assert key[1] == level
