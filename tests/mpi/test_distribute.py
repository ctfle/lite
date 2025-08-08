import numpy as np
import pytest

from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.distribute import Distributor

# to test run pytest with the additional flag --with-mpi


class TestDistributeTasks:
    @pytest.fixture(scope="function")
    def random_lattice(self, request):
        size, level = request.param
        random_matrix_dict = {
            (n, level): np.random.uniform(size=(2 ** (level + 1), 2 ** (level + 1)))
            for n in range(size)
        }
        return LatticeDict.from_dict(random_matrix_dict)

    @pytest.mark.parametrize(
        "random_lattice, level",
        [((10, 1), 1), ((11, 2), 2), ((23, 3), 3), ((26, 4), 4)],
        indirect=["random_lattice"],
    )
    def test_ordered_keys(self, random_lattice, level):
        distributor = Distributor(random_lattice, level)
        ordered_keys = distributor._ordered_keys

        for index, key in enumerate(random_lattice.keys()):
            assert key == ordered_keys[index]

    @pytest.mark.parametrize(
        "random_lattice, number_of_splits, shift, level",
        [
            ((10, 1), 4, 1, 1),
            ((10, 1), 4, 0, 1),
            ((10, 1), 5, 1, 1),
            ((10, 1), 5, 0, 1),
            ((11, 1), 4, 1, 1),
            ((11, 1), 4, 0, 1),
            ((11, 1), 5, 1, 1),
        ],
        indirect=["random_lattice"],
    )
    def test_split_up_keys(self, random_lattice, number_of_splits, shift, level):
        """Test that the shifting works correctly"""
        distributor = Distributor(random_lattice, level)
        split_up_keys = distributor._split_keys(
            number_of_splits=number_of_splits, shift=shift
        )
        for k, key_block in enumerate(split_up_keys[:-1]):
            next_key_block = split_up_keys[k + 1]
            for s in range(shift):
                assert key_block[-(s + 1)] == next_key_block[s]

    @pytest.mark.parametrize(
        "random_lattice, number_of_splits, shift, level",
        [((10, 1), 19, 0, 1), ((10, 1), 19, 1, 1), ((31, 1), 40, 1, 1)],
        indirect=["random_lattice"],
    )
    def test_split_with_too_many_splits(
        self, random_lattice, number_of_splits, shift, level
    ):
        """
        We use too many splits for a short list.
        This should result in trailing empty lists.
        """
        lattice_length = len(random_lattice)
        distributor = Distributor(random_lattice, level)
        split_up_keys = distributor._split_keys(
            number_of_splits=number_of_splits, shift=shift
        )
        # check that shifting works fine regardless
        for k, key_block in enumerate(split_up_keys[: (lattice_length - 1)]):
            next_key_block = split_up_keys[k + 1]
            for s in range(shift):
                assert key_block[-(s + 1)] == next_key_block[s]

        # check for trailing empty lists
        for key_block in split_up_keys[lattice_length:]:
            assert key_block == []

    @pytest.mark.parametrize(
        "random_lattice, number_of_splits, shift, level",
        [
            ((10, 1), 19, 0, 2),
            ((10, 1), 19, 1, 2),
            ((31, 2), 40, 1, 3),
            ((31, 2), 40, 1, 1),
        ],
        indirect=["random_lattice"],
    )
    def test_no_density_matrices_at_level(
        self, random_lattice, number_of_splits, shift, level
    ):
        """
        No density matrices are present at the specified level.
        This triggers an assertion.
        """
        with pytest.raises(AssertionError):
            _ = Distributor(random_lattice, level)

    @pytest.mark.parametrize(
        "size, level, number_of_splits",
        [(0, 0, 5), (0, 1, 10), (0, 1, 7), (51, 3, 0)],
    )
    def test_distribute_empty_lattice(self, size, level, number_of_splits):
        """Try to distribute an empty lattice"""
        with pytest.raises(AssertionError):
            distributor = Distributor(LatticeDict(), level)
            _ = distributor.distribute(number_of_workers=number_of_splits, shift=1)

    @pytest.mark.parametrize(
        "random_lattice, number_of_splits, shift, level",
        [
            ((3, 0), 2, 0, 0),
            ((25, 1), 3, 1, 1),
            ((25, 1), 4, 0, 1),
            ((26, 2), 2, 1, 2),
            ((26, 2), 3, 0, 2),
            ((26, 2), 4, 1, 2),
            ((51, 3), 2, 0, 3),
            ((51, 3), 5, 1, 3),
            ((111, 2), 2, 0, 2),
            ((111, 2), 20, 1, 2),
            ((111, 2), 21, 0, 2),
            ((111, 2), 50, 1, 2),
        ],
        indirect=["random_lattice"],
    )
    def test_distribute_lattice(self, random_lattice, number_of_splits, shift, level):
        """Test that distribute works well with and without shift"""

        distributor = Distributor(random_lattice, level)
        distributed_lattice = distributor.distribute(number_of_workers=number_of_splits)
        # check that shifting works fine

        for l, lattice in enumerate(distributed_lattice[:-1]):
            next_lattice = distributed_lattice[l + 1]
            this_lattice_keys = list(lattice.keys())
            next_lattice_keys = list(next_lattice.keys())
            for s in range(shift):
                this_key = this_lattice_keys[-(s + 1)]
                next_key = next_lattice_keys[s]
                assert np.allclose(lattice[this_key], next_lattice[next_key])

    @pytest.mark.parametrize(
        "random_lattice, number_of_splits, shift, level",
        [
            ((3, 0), 5, 0, 0),
            ((25, 0), 32, 1, 0),
            ((25, 1), 31, 0, 1),
            ((26, 2), 35, 1, 2),
            ((26, 2), 34, 0, 2),
        ],
        indirect=["random_lattice"],
    )
    def test_distribute_lattice_too_many_splits(
        self, random_lattice, number_of_splits, shift, level
    ):
        """Test that distribute works well even if we have too manz splits"""

        lattice_length = len(random_lattice)
        distributor = Distributor(random_lattice, level)
        distributed_lattice = distributor.distribute(number_of_workers=number_of_splits)
        # check that shifting works fine
        for ind, lattice in enumerate(distributed_lattice[: (lattice_length - 1)]):
            next_lattice = distributed_lattice[ind + 1]
            this_lattice_keys = list(lattice.keys())
            next_lattice_keys = list(next_lattice.keys())
            for s in range(shift):
                this_key = this_lattice_keys[-(s + 1)]
                next_key = next_lattice_keys[s]
                assert np.allclose(lattice[this_key], next_lattice[next_key])

        # check for trailing empty lists
        for lattice in distributed_lattice[lattice_length:]:
            assert lattice == LatticeDict()
