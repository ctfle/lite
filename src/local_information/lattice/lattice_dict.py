from __future__ import annotations

import logging
from copy import deepcopy
from itertools import compress
from numbers import Number
from typing import ItemsView, Iterator, Union, Type, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from local_information.lattice.protocols import Arithmetics
from local_information.mpi.mpi_funcs import get_mpi_variables

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()

logger = logging.getLogger()
Matrix = Union[np.ndarray, sparse.csr_matrix, sparse.csc_matrix, list, tuple]


class LatticeDict(dict):
    """!
    A class that extends functionalities of dict to compute and handle the information lattice.
    'LatticeDict's can be added and scalar multiplied. If the values are not numeric, they must
    be of the same type that allows arithmetic operations.
    """

    def __init__(
        self,
        items: ItemsView[tuple, ArrayLike] | Iterator[tuple, ArrayLike] = iter(dict()),
    ):
        super().__init__()
        types = []
        for key, value in items:
            if not isinstance(key, tuple):
                raise ValueError

            if not isinstance(value, Number):
                if not isinstance(value, Arithmetics):
                    raise ValueError(
                        f"{type(value)} does not satisfy required arithmetics"
                        f" (addition, subtraction and scalar multiplication)"
                    )
                types += [type(value)]
            self[key] = value

        if not all((x is types[0]) for x in types):
            raise TypeError("inhomogeneous types")

    @property
    def _type(self) -> None | Type:
        keys = list(self.keys())
        if keys:
            value = self[keys[0]]
            return type(value)
        else:
            return None

    @property
    def _is_numeric(self) -> bool:
        keys = list(self.keys())
        if keys:
            value = self[keys[0]]
            return isinstance(value, Number)
        else:
            return False

    @classmethod
    def from_list(cls, keys: list[tuple], values: Sequence[ArrayLike]) -> LatticeDict:
        return cls(zip(keys, values))

    @classmethod
    def from_dict(cls, input_dict: dict[tuple, ArrayLike]) -> LatticeDict:
        return cls(input_dict.items())

    def __add__(self, other: LatticeDict) -> LatticeDict:
        numeric_or_empty = self._type is None or other._type is None
        if self._type != other._type and not numeric_or_empty:
            raise TypeError(
                "cannot add {} and {} objects".format(self._type, other._type)
            )

        sum_dict = LatticeDict()
        for self_key, self_value in self.items():
            if other.get(self_key) is None:
                sum_dict[self_key] = self_value
            else:
                sum_dict[self_key] = self_value + other[self_key]

        for other_key, other_value in other.items():
            if self.get(other_key) is None:
                sum_dict[other_key] = other_value

        return sum_dict

    def __mul__(self, other: Number) -> LatticeDict:
        """scalar multiplication"""
        if isinstance(other, Number):
            result = LatticeDict()
            for key in list(self.keys()):
                result[key] = self[key] * other
        else:
            raise ValueError

        return result

    def __rmul__(self, other: Number) -> LatticeDict:
        return self.__mul__(other)

    def __sub__(self, other: LatticeDict) -> LatticeDict:
        return self.__add__(other.__mul__(-1))

    # TODO add a bound to ensure allclose can be used
    def __eq__(self, other: LatticeDict) -> bool:
        if not isinstance(other, LatticeDict):
            raise TypeError

        # compare keys and values
        if len(self.keys()) != len(other.keys()):
            return False

        for key, val in self.items():
            if not np.allclose(val, other[key]):
                return False
        return True

    def merge(self, other: LatticeDict):
        """!
        Merges two lattice dicts. This means it adds all the key-value pairs
        which are not in self but in other
        """
        for key in other:
            if key in self:
                continue
            else:
                self[key] = other[key]

    def overlap(self, other: LatticeDict) -> list[tuple]:
        """!
        computes the overlap with 'other'
        :returns: the corresponding keys
        """
        overlap = []
        for key in self.keys():
            if key in other:
                overlap += [key]
        return overlap

    def __setitem__(self, key: tuple, value):
        if not isinstance(key, tuple):
            raise TypeError("key must be tuple")

        if not isinstance(value, Arithmetics):
            raise TypeError(
                f"incompatible type: {type(value)} does not satisfy arithmetics"
            )

        if self._type is None:
            super().__setitem__(key, value)
        else:
            if self._is_numeric and isinstance(value, Number):
                super().__setitem__(key, value)
            else:
                if self._type == type(value):
                    super().__setitem__(key, value)
                else:
                    raise TypeError(
                        "wrong data type: type is {} but {} was given",
                        self._type,
                        type(value),
                    )

    def smallest_at_level(self, ell: int) -> float | None:
        n_list = self.n_at_level(ell)
        if n_list:
            return min(n_list)
        else:
            return None

    def largest_at_level(self, ell: int) -> float | None:
        n_list = self.n_at_level(ell)
        if n_list:
            return max(n_list)
        else:
            return None

    def dim_at_level(self, ell: int) -> int:
        return len(self.n_at_level(ell))

    def boundaries(self, ell: int) -> tuple[float, float]:
        """!
        get the boundary keys at level ell
        """
        n_max = self.largest_at_level(ell)
        n_min = self.smallest_at_level(ell)
        return n_min, n_max

    def get_and_broadcast_boundaries(self, ell: int):
        n_min, n_max = None, None
        if RANK == 0:
            n_min, n_max = self.boundaries(ell)

        n_min = COMM.bcast(n_min, root=0)
        n_max = COMM.bcast(n_max, root=0)

        return n_min, n_max

    def n_at_level(self, ell: int) -> list:
        return [n for (n, level) in self.keys() if level == ell]

    def keys_at_level(self, ell: int) -> LatticeDictIterator:
        return LatticeDictIterator(self, ell, values=False)

    def values_at_level(self, ell: int) -> LatticeDictIterator:
        return LatticeDictIterator(self, ell, keys=False)

    def items_at_level(self, ell) -> LatticeDictIterator:
        return LatticeDictIterator(self, ell)

    def get_max_level(self) -> int:
        return max(map(lambda x: x[1], self.keys()))

    def has_single_entry_at_level(self, level: int) -> bool:
        return np.allclose(self.largest_at_level(level), self.smallest_at_level(level))

    def add_sites(self, delta_n: int, TI_keys: list[tuple], orientation: str = "right"):
        """!
        Adds delta_n sites at the end given by orientation
        """
        size = len(TI_keys)
        TI_keys = sorted(TI_keys)

        check = True
        r = 0
        if orientation == "left":
            n_min = min(map(lambda x: x[0], TI_keys))
            while check:
                for TI_k in TI_keys[::-1]:
                    TI_n = TI_k[0]
                    if (TI_n - r * size) >= (n_min - delta_n):
                        mod_key = (TI_k[0] - r * size, TI_k[1])
                        self[mod_key] = self[TI_k]
                    else:
                        check = False
                r += 1
        if orientation == "right":
            n_max = max(map(lambda x: x[0], TI_keys))
            while check:
                for TI_k in TI_keys:
                    TI_n = TI_k[0]
                    if (TI_n + r * size) <= (n_max + delta_n):
                        mod_key = (TI_k[0] + size * r, TI_k[1])
                        self[mod_key] = self[TI_k]
                    else:
                        check = False
                r += 1
        pass

    def delete_sites(self, delta_n: int, ell: int, orientation: str = "right"):
        """!
        Deletes delta_n sites at the right or left end
        """
        if orientation == "left":
            n_min = self.smallest_at_level(ell)
            for n in range(delta_n):
                self.pop((n_min + n, ell), None)
        elif orientation == "right":
            n_max = self.largest_at_level(ell)
            for n in range(delta_n):
                self.pop((n_max - n, ell), None)
        pass

    def kill_all_except(self, ell: int):
        """!
        Delete all key-value pairs except those where key[1]==ell
        """
        for key in list(self.keys()):
            if key[1] != ell:
                self.pop(key, None)
        pass

    def dagger(self) -> LatticeDict:
        daggered = self.deepcopy()
        for key in daggered.keys():
            daggered[key] = np.conjugate(np.transpose(daggered[key]))

        return daggered

    def to_array(self) -> LatticeDict:
        if self._type == sparse.csr_matrix or self._type == sparse.csc_matrix:
            array_dict = LatticeDict()
            for key, val in self.items():
                array_dict[key] = val.toarray()
            return array_dict
        else:
            return self

    def deepcopy(self):
        return deepcopy(self)


class LatticeDictIterator:
    def __init__(
        self,
        lattice: LatticeDict,
        level: int | None = None,
        keys: bool = True,
        values: bool = True,
    ):
        self.tags = (keys, values)
        self.level = level
        self.lattice = lattice
        self.n = None
        self.reset()

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.n is not None:
            self.n += 1

        if self.n is not None and self.n <= self.lattice.largest_at_level(self.level):
            key = (self.n, self.level)
            next_data = tuple(compress((key, self.lattice.get(key)), self.tags))
            if len(next_data) == 1:
                return next_data[0]
            else:
                return next_data
        else:
            raise StopIteration

    def reset(self):
        if self.level is None or self.lattice.smallest_at_level(self.level) is None:
            self.n = None
        else:
            self.n = self.lattice.smallest_at_level(self.level) - 1
        pass


class LatticeDictKey:
    def __init__(self, n: float, level: int):
        self.n = n
        self.level = level

    @classmethod
    def from_tuple(cls, key: tuple[float, int]):
        return cls(n=key[0], level=key[1])

    def __eq__(self, other):
        if not isinstance(other, LatticeDictKey):
            return False

        return self.n == other.n and self.level == other.level

    def __hash__(self):
        return hash((self.n, self.level))
