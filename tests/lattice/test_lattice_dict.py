import numpy as np
import pytest

from local_information.lattice.lattice_dict import LatticeDict


class SomeArithmetic:
    def __add__(self, other):
        return other

    def __sub__(self, other):
        return other

    def __mul__(self, other):
        return


class OtherArithmetic:
    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, other):
        return self


class NonArithmetic:
    def __add__(self, other):
        return self


class TestLatticeDict:
    @pytest.fixture
    def test_lattice_dict1(self):
        vals = [
            np.array(
                [[0.997, 0, 0, 0], [0, 0.001, 0, 0], [0, 0, 0.001, 0], [0, 0, 0, 0.001]]
            )
        ]
        keys = [(1, 1)]
        return LatticeDict.from_list(keys, vals)

    @pytest.fixture
    def test_lattice_dict2(self):
        vals = [
            np.array(
                [[0.001, 0, 0, 0], [0, 0.997, 0, 0], [0, 0, 0.001, 0], [0, 0, 0, 0.001]]
            )
        ]
        keys = [(1, 2)]
        return LatticeDict.from_list(keys, vals)

    @pytest.fixture
    def test_lattice_dict3(self):
        vals = [
            np.array(
                [[0.001, 0, 0, 0], [0, 0.997, 0, 0], [0, 0, 0.001, 0], [0, 0, 0, 0.001]]
            ),
            np.array(
                [[0.001, 0, 0, 0], [0, 0.997, 0, 0], [0, 0, 0.001, 0], [0, 0, 0, 0.001]]
            ),
        ]
        keys = [(1, 1), (1, 2)]
        return LatticeDict.from_list(keys, vals)

    @pytest.fixture
    def test_lattice_dict4(self):
        vals = [1.2, 3.4, 5.6, 6.7, 0.0, 1j, 1, 1, 1]
        keys = [
            (1, 2),
            (2, 3),
            (4, 5),
            (4, 6),
            (3, 6),
            (10, 20),
            (11, 7),
            (12, 7),
            (13, 7),
        ]
        return LatticeDict.from_list(keys, vals)

    def test_lattice_dict_sum(self, test_lattice_dict1, test_lattice_dict2):
        sum_dict = test_lattice_dict1 + test_lattice_dict2
        sum_keys = set(test_lattice_dict1.keys()).union(set(test_lattice_dict2.keys()))
        assert set(sum_dict.keys()) == sum_keys

        sum_of_identical_dicts = test_lattice_dict1 + test_lattice_dict1
        for key, val in sum_of_identical_dicts.items():
            assert np.allclose(val, 2 * test_lattice_dict1[key])

    def test_lattice_dict_subtraction(self, test_lattice_dict1, test_lattice_dict2):
        subtract_dict = test_lattice_dict1 - test_lattice_dict2
        subtract_keys = set(test_lattice_dict1.keys()).union(
            set(test_lattice_dict2.keys())
        )
        assert set(subtract_dict.keys()) == subtract_keys

        subtract_of_identical_dicts = test_lattice_dict1 - test_lattice_dict1
        for key, val in subtract_of_identical_dicts.items():
            assert np.allclose(val, np.zeros((4, 4)))

    def test_scalar_multiplication(
        self,
        test_lattice_dict1,
        test_lattice_dict2,
        test_lattice_dict3,
        test_lattice_dict4,
    ):
        combined = test_lattice_dict1 + test_lattice_dict2
        multiplied = 2.123 * combined
        for key, val in multiplied.items():
            assert np.allclose(val, 2.123 * combined[key])

        multiplied = 3.14 * test_lattice_dict3
        for key, val in multiplied.items():
            test_dict_array = test_lattice_dict3[key]
            val_array = val
            assert np.allclose(val_array, 3.14 * test_dict_array)

        multiplied = 5.678 * test_lattice_dict4
        for key, val in multiplied.items():
            assert np.allclose(val, 5.678 * test_lattice_dict4[key])

    def test_smallest_at_level(self, test_lattice_dict4):
        assert test_lattice_dict4.smallest_at_level(1) is None
        assert test_lattice_dict4.smallest_at_level(2) == 1
        assert test_lattice_dict4.smallest_at_level(6) == 3

    def test_largest_at_level(self, test_lattice_dict4):
        assert test_lattice_dict4.largest_at_level(1) is None
        assert test_lattice_dict4.largest_at_level(2) == 1
        assert test_lattice_dict4.largest_at_level(6) == 4

    def test_keys_at_level(self, test_lattice_dict4):
        assert list(test_lattice_dict4.keys_at_level(1)) == []
        level3 = list(test_lattice_dict4.keys_at_level(3))
        assert len(level3) == 1
        assert level3[0][0] == 2
        assert level3[0][1] == 3
        level6 = list(test_lattice_dict4.keys_at_level(6))
        assert len(level6) == 2
        assert (3, 6) in level6
        assert (4, 6) in level6

    def test_merge(self, test_lattice_dict1, test_lattice_dict3):
        keys1 = test_lattice_dict1.keys()
        test_lattice_dict1.merge(test_lattice_dict3)
        keys3 = test_lattice_dict3.keys()
        for key, val in test_lattice_dict1.items():
            assert key in keys1 or key in keys3

        for key, val in test_lattice_dict1.items():
            assert key in set(keys1).union(set(keys3))

    def test_add_sites(self, test_lattice_dict4):
        test_lattice_dict4.add_sites(2, TI_keys=[(1, 2)], orientation="right")
        assert len(list(test_lattice_dict4.keys_at_level(2))) == 3
        for key in test_lattice_dict4.keys_at_level(2):
            assert test_lattice_dict4[key] == 1.2
            assert key[0] in [1, 2, 3]

        test_lattice_dict4.add_sites(2, TI_keys=[(1, 2)], orientation="left")
        assert len(list(test_lattice_dict4.keys_at_level(2))) == 5
        for key in test_lattice_dict4.keys_at_level(2):
            assert test_lattice_dict4[key] == 1.2
            assert key[0] in [-1, 0, 1, 2, 3]

    def test_delete_sites(self, test_lattice_dict4):
        initial_keys = list(test_lattice_dict4.keys_at_level(2))
        test_lattice_dict4.add_sites(2, TI_keys=[(1, 2)], orientation="right")
        test_lattice_dict4.add_sites(2, TI_keys=[(1, 2)], orientation="left")
        test_lattice_dict4.delete_sites(delta_n=2, ell=2, orientation="right")
        test_lattice_dict4.delete_sites(delta_n=2, ell=2, orientation="left")

        for key in test_lattice_dict4.keys_at_level(2):
            assert key in initial_keys
            assert test_lattice_dict4[key] == 1.2

        assert len(list(test_lattice_dict4.keys_at_level(2))) == len(list(initial_keys))

    def test_kill_all_except(self, test_lattice_dict4):
        test_lattice_dict4.kill_all_except(6)
        assert len(test_lattice_dict4.keys()) == 2

    def test_dagger(self, test_lattice_dict1, test_lattice_dict4):
        values = test_lattice_dict4.values()

        daggered_lattice = test_lattice_dict4.dagger()
        for key, val in daggered_lattice.items():
            if key[0] == 10 and key[1] == 20:
                assert val == -1j
            else:
                assert val in values

        daggered_lattice = test_lattice_dict1.dagger()
        for key, val in daggered_lattice.items():
            assert np.allclose(val, test_lattice_dict1[key])

    def test_items_at_level(self, test_lattice_dict4):
        keys = test_lattice_dict4.keys_at_level(7)
        for key, val in test_lattice_dict4.items_at_level(7):
            assert key in keys
            assert val == 1

    def test_eq(self, test_lattice_dict4, test_lattice_dict3, test_lattice_dict2):
        assert test_lattice_dict4 == test_lattice_dict4
        assert test_lattice_dict3 == test_lattice_dict3

        assert test_lattice_dict3 != test_lattice_dict4
        assert test_lattice_dict3 != test_lattice_dict2
        assert test_lattice_dict4 != test_lattice_dict2

    def test_from_dict(self):
        test_dict = {(1, 0): np.array([1, 2, 3]), (1, 1): np.array([4, 5, 6])}
        lattice = LatticeDict.from_dict(test_dict)
        for key, value in lattice.items():
            assert np.allclose(test_dict[key], value)
        pass

    def test_from_list(self):
        keys = [(1, 0), (1, 1)]
        values = [1.2, 3.4]
        lattice = LatticeDict.from_list(keys, values)
        for key, value in lattice.items():
            assert key in keys
            assert value in values
        pass

    def test_add(self, test_lattice_dict2, test_lattice_dict3):
        summed_dict = test_lattice_dict3 + test_lattice_dict2
        overlap = test_lattice_dict2.overlap(test_lattice_dict2)
        for key in overlap:
            assert np.allclose(
                summed_dict[key], test_lattice_dict2[key] + test_lattice_dict3[key]
            )
        pass

    def test_add_to_empty(self, test_lattice_dict3):
        summed_dict = LatticeDict()
        summed_dict += test_lattice_dict3
        for key, value in summed_dict.items():
            assert np.allclose(value, test_lattice_dict3[key])
        pass

    def test_settitem(self):
        test_dict = LatticeDict()
        for i in range(10):
            test_dict[(i, i + 1)] = 1.2 + i
        assert test_dict[(0, 1)] == 1.2
        assert test_dict[(9, 10)] == 1.2 + 9

        test_dict = LatticeDict()
        try:
            for i in range(10):
                test_dict[i] = 1
        except TypeError:
            pass

        test_dict = LatticeDict()
        try:
            for i in range(10):
                test_dict[(i, i)] = "tests"
        except TypeError:
            pass

    def test_arithemtic_class(self):
        test_dict = LatticeDict()
        assert test_dict._is_numeric is False
        assert test_dict._type is None

        for i in range(10):
            test_dict[(i, i)] = SomeArithmetic()
        assert test_dict._type == SomeArithmetic

    def test_other_arithemtic_class(self):
        test_dict = LatticeDict()
        k = 10
        for i in range(k):
            test_dict[(i, i)] = SomeArithmetic()
        assert test_dict._type == SomeArithmetic
        try:
            test_dict[(k, k)] = OtherArithmetic()
            assert False
        except TypeError:
            pass

    def test_non_arithemtic_class(self):
        test_dict = LatticeDict()
        try:
            test_dict[(1, 2)] = NonArithmetic()
            assert False
        except TypeError:
            pass


class TestLatticeDictIterator:
    @pytest.fixture
    def test_lattice_dict(self) -> tuple[LatticeDict, list, list]:
        vals = [np.random.normal(size=(4, 4)) for _ in range(10)]
        keys = [(n, 1) for n in range(10)]
        return LatticeDict.from_list(keys, vals), vals, keys

    def test_values_at_level(self, test_lattice_dict):
        lattice, values, _ = test_lattice_dict
        for v, value in enumerate(lattice.values_at_level(1)):
            assert np.allclose(values[v], value)
        pass

    def test_keys_at_level(self, test_lattice_dict):
        lattice, _, keys = test_lattice_dict
        for k, key in enumerate(lattice.keys_at_level(1)):
            assert key == keys[k]
        pass

    def test_items_at_level(self, test_lattice_dict):
        lattice, values, keys = test_lattice_dict
        for k, (key, value) in enumerate(lattice.items_at_level(1)):
            assert key == keys[k]
            assert np.allclose(values[k], value)
        pass

    def test_reuse(self, test_lattice_dict):
        lattice, values, keys = test_lattice_dict

        keys_iterator = lattice.keys_at_level(1)
        values_iterator = lattice.values_at_level(1)
        items_iterator = lattice.items_at_level(1)
        # loop once
        for k, key in enumerate(keys_iterator):
            assert keys[k] == key
        # loop again
        for k, key in enumerate(keys_iterator):
            assert keys[k] == key

        for v, value in enumerate(values_iterator):
            assert np.allclose(value, values[v])
        for v, value in enumerate(values_iterator):
            assert np.allclose(value, values[v])

        for k, (key, value) in enumerate(items_iterator):
            assert key == keys[k]
            assert np.allclose(values[k], value)
        for k, (key, value) in enumerate(items_iterator):
            assert key == keys[k]
            assert np.allclose(values[k], value)

        pass

    def test_empty(self, test_lattice_dict):
        lattice, _, _ = test_lattice_dict
        for _ in lattice.items_at_level(2):
            assert False
        for _ in lattice.keys_at_level(2):
            assert False
        for _ in lattice.values_at_level(2):
            assert False
        pass
