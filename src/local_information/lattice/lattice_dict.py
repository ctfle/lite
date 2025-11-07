from __future__ import annotations

import logging
from copy import deepcopy
from functools import cached_property
from itertools import compress
from numbers import Number
from typing import ItemsView, Iterator, Iterable, Union, Type, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from local_information.lattice.protocols import Arithmetics

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
        items: ItemsView[LatticeKey, ArrayLike]
        | Iterator[LatticeKey, ArrayLike] = iter(dict()),
    ):
        super().__init__()
        types = []
        for key, value in items:
            if not isinstance(key, LatticeKey):
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
    def from_list(
        cls, keys: list[LatticeKey], values: Sequence[ArrayLike]
    ) -> LatticeDict:
        return cls(zip(keys, values))

    @classmethod
    def from_dict(cls, input_dict: dict[LatticeKey, ArrayLike]) -> LatticeDict:
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

    def overlap(self, other: LatticeDict) -> list[LatticeKey]:
        """!
        computes the overlap with 'other'
        :returns: the corresponding keys
        """
        overlap = []
        for key in self.keys():
            if key in other:
                overlap += [key]
        return overlap

    def __setitem__(self, key: LatticeKey, value):
        if not isinstance(key, LatticeKey):
            raise TypeError("key must be LatticeKey")

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

    def smallest_at_level(self, level: int) -> float | None:
        n_list = self.coords_at_level(level)
        if n_list:
            return min(n_list)
        else:
            return None

    def largest_at_level(self, level: int) -> float | None:
        n_list = self.coords_at_level(level)
        if n_list:
            return max(n_list)
        else:
            return None

    def leftmost_key_at_level(self, level: int):
        n_list = self.coords_at_level(level)
        if n_list:
            return LatticeKey(min(n_list), level)
        else:
            return None

    def rightmost_key_at_level(self, level: int):
        n_list = self.coords_at_level(level)
        if n_list:
            return LatticeKey(max(n_list), level)
        else:
            return None

    def dim_at_level(self, ell: int) -> int:
        return len(self.coords_at_level(ell))

    def boundaries(self, ell: int) -> tuple[float, float]:
        """!
        get the boundary keys at level ell
        """
        n_max = self.largest_at_level(ell)
        n_min = self.smallest_at_level(ell)
        return n_min, n_max

    def coords_at_level(self, ell: int) -> list[float]:
        return [key.coord for key in self.keys() if key.level == ell]

    def keys_at_level(self, ell: int) -> LatticeDictIterator:
        return LatticeDictIterator(self, ell, values=False)

    def values_at_level(self, ell: int) -> LatticeDictIterator:
        return LatticeDictIterator(self, ell, keys=False)

    def items_at_level(self, ell) -> LatticeDictIterator:
        return LatticeDictIterator(self, ell)

    def get_max_level(self) -> int:
        return max(map(lambda x: x.level, self.keys()))

    def has_single_entry_at_level(self, level: int) -> bool:
        return np.allclose(self.largest_at_level(level), self.smallest_at_level(level))

    def kill_all_except(self, ell: int):
        """!
        Delete all key-value pairs except those where key.level==ell
        """
        for key in list(self.keys()):
            if key.level != ell:
                self.pop(key, None)

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

    @cached_property
    def _largest_at_level(self):
        return self.lattice.largest_at_level(self.level)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.n is not None:
            self.n += 1

        if self.n is not None and self.n <= self._largest_at_level:
            key = LatticeKey(self.n, self.level)
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


class LatticeKey:
    def __init__(self, coord: float, level: int, name: str | None = None):
        self.coord = coord
        self.level = level
        self.name = name
        assert isinstance(self.level, int)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        if isinstance(value, int):
            self._level = value
        elif isinstance(value, float):
            if value.is_integer():  # checks if float is like 1.0, 2.0 etc
                self._level = int(value)
            else:
                raise ValueError(
                    f"Level must be int or float with no fractional part, got {value}"
                )

    @classmethod
    def from_tuple(cls, key: tuple[float, int], name: str | None = None) -> LatticeKey:
        return cls(coord=key[0], level=key[1], name=name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, LatticeKey):
            return False
        if self.name:
            return (
                np.allclose(self.coord, other.coord)
                and self.level == other.level
                and self.name == other.name
            )
        else:
            return np.allclose(self.coord, other.coord) and self.level == other.level

    def __hash__(self):
        if self.name:
            return hash((self.coord, self.level, self.name))
        else:
            return hash((self.coord, self.level))

    def __str__(self) -> str:
        if self.name:
            return f"({self.coord}, {self.level}, {self.name})"
        else:
            return f"({self.coord}, {self.level})"

    __repr__ = __str__

    def get_lower_level_left(self) -> LatticeKey:
        assert self.level != 0, "level is 0, no lower level existing"
        return LatticeKey(self.coord - 0.5, self.level - 1)

    def get_lower_level_right(self) -> LatticeKey:
        assert self.level != 0, "level is 0, no lower level existing"
        return LatticeKey(self.coord + 0.5, self.level - 1)

    def get_higher_level_right(self) -> LatticeKey:
        return LatticeKey(self.coord + 0.5, self.level + 1)

    def get_higher_level_left(self) -> LatticeKey:
        return LatticeKey(self.coord - 0.5, self.level + 1)


def keys_from_iterable(keys: Iterable[tuple[float, int]]):
    return list(map(lambda x: LatticeKey.from_tuple(x), keys))
