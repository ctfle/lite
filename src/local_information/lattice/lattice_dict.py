from __future__ import annotations

import logging
from copy import deepcopy
from functools import cached_property
from itertools import compress
from numbers import Number
from typing import (
    ItemsView,
    Iterator,
    Iterable,
    Union,
    Type,
    Sequence,
    TypeVar,
    Generic,
    Any,
)

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from local_information.lattice.protocols import Arithmetics

logger = logging.getLogger()
Matrix = Union[np.ndarray, sparse.csr_matrix, sparse.csc_matrix, list, tuple]

T = TypeVar("T")


class LatticeDict(dict, Generic[T]):
    """
    A class that extends functionalities of dict to compute and handle the information lattice.
    'LatticeDict's can be added and scalar multiplied. If the values are not numeric, they must
    be of the same type that allows arithmetic operations.
    """

    def __init__(
        self,
        items: ItemsView[LatticeKey, ArrayLike] | Iterator[LatticeKey, ArrayLike] = (),
    ):
        super().__init__()

        self._value_type: Type | None = None

        for key, value in items:
            self._validate_and_set_item(key, value)

    def _validate_and_set_item(self, key: Any, value: Any) -> None:
        key = self._check_key(key)
        self._validate_value(value)
        super().__setitem__(key, value)

    def _validate_value(self, value: Any) -> None:
        self._check_value_arithmetics(value)
        self._check_value_type_consistency(value)

    @staticmethod
    def _check_value_arithmetics(value: Any) -> None:
        if isinstance(value, Number):
            return
        if isinstance(value, Arithmetics):
            return
        raise TypeError(
            f"Value {value!r} of type {type(value)!r} does not support required "
            "arithmetic operations (addition, subtraction, scalar multiplication)"
        )

    def _check_value_type_consistency(self, value: Any) -> None:
        self._ensure_value_type_attr()
        if self._value_type is None:
            # First item defines the type of this lattice
            self._value_type = type(value)
            return

        if isinstance(value, Number) and issubclass(self._value_type, Number):
            # All numeric types allowed together
            return

        if type(value) is not self._value_type:
            raise TypeError(
                f"Inhomogeneous value type: expected {self._value_type!r}, "
                f"got {type(value)!r}"
            )

    @staticmethod
    def _check_key(key: Any) -> LatticeKey:
        if not isinstance(key, LatticeKey):
            raise TypeError(f"Key {key!r} has type {type(key)!r}, expected LatticeKey")
        return key

    def _check_value_compatibility(self, other: LatticeDict):
        compatible = True
        if self._value_type is not None and other._value_type is not None:
            if self._value_type is not other._value_type:
                compatible = issubclass(self._value_type, Number) and issubclass(
                    other._value_type, Number
                )
        return compatible

    def _ensure_value_type_attr(self) -> None:
        # This must *not* assume __init__ has run.
        # Guard is necessary since we use MPI where we serialise objects using pickle
        # which are not calling the init but LatticeDict.__new__(LatticeDict) and uses
        # __setitem__ for each key value pair. To ensure _value_type is set we use the guard
        # when calling __setitem__
        if not hasattr(self, "_value_type"):
            self._value_type = None

    @classmethod
    def from_list(
        cls, keys: list[LatticeKey], values: Sequence[ArrayLike]
    ) -> LatticeDict:
        return cls(zip(keys, values))

    @classmethod
    def from_dict(cls, input_dict: dict[LatticeKey, ArrayLike]) -> LatticeDict:
        return cls(input_dict.items())

    def to_dict(self) -> dict[tuple[float, int], ArrayLike]:
        output = dict()
        for key, value in self.items():
            output[key.to_tuple()] = value
        return output

    def __add__(self, other: LatticeDict[T]) -> LatticeDict[T]:
        if not self._check_value_compatibility(other):
            raise TypeError(
                "cannot add {} and {} objects".format(
                    self._value_type, other._value_type
                )
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

    def __mul__(self, value: Number) -> LatticeDict[T]:
        """scalar multiplication"""
        if isinstance(value, Number):
            result = LatticeDict()
            for key in list(self.keys()):
                result[key] = self[key] * value
        else:
            raise ValueError

        return result

    def __rmul__(self, value: Number) -> LatticeDict:
        return self.__mul__(value)

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
        """
        Merges two lattice dicts. This means it adds all the key-value pairs
        which are not in self but in other. Note: ignores existing key value pairs.
        Existing pairs will not be updated!
        """
        if not isinstance(other, LatticeDict):
            raise ValueError("Can only merge LatticeDicts")
        for key in other:
            if key in self:
                continue
            else:
                self[key] = other[key]

    def overlap(self, other: LatticeDict) -> list[LatticeKey]:
        """
        computes the overlap with 'other'
        :returns: the corresponding keys
        """
        overlap = []
        for key in self.keys():
            if key in other:
                overlap += [key]
        return overlap

    def __setitem__(self, key: LatticeKey, value: Any):
        self._validate_and_set_item(key, value)

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

    def boundaries(self, level: int) -> tuple[float, float]:
        """
        Get the boundary keys at level
        """
        n_max = self.largest_at_level(level)
        n_min = self.smallest_at_level(level)
        return n_min, n_max

    def coords_at_level(self, level: int) -> list[float]:
        return [key.coord for key in self.keys() if key.level == level]

    def keys_at_level(self, level: int) -> LatticeDictIterator:
        return LatticeDictIterator(self, level, values=False)

    def values_at_level(self, level: int) -> LatticeDictIterator[T]:
        return LatticeDictIterator(self, level, keys=False)

    def items_at_level(self, level: int) -> LatticeDictIterator[T]:
        return LatticeDictIterator(self, level)

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

    def drop_boundaries(self, level: int):
        """Drops the boundary density matrices at given level."""
        n_min, n_max = self.boundaries(level)
        self.pop(LatticeKey(n_max, level), None)
        self.pop(LatticeKey(n_min, level), None)

    def dagger(self) -> LatticeDict:
        daggered = self.deepcopy()
        for key in daggered.keys():
            daggered[key] = np.conjugate(np.transpose(daggered[key]))

        return daggered

    def to_array(self) -> LatticeDict:
        if (
            self._value_type == sparse.csr_matrix
            or self._value_type == sparse.csc_matrix
        ):
            array_dict = LatticeDict()
            for key, val in self.items():
                array_dict[key] = val.toarray()
            return array_dict
        else:
            return self

    def deepcopy(self):
        return deepcopy(self)


class LatticeDictIterator(Iterator):
    def __init__(
        self,
        lattice: LatticeDict[T],
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

    def __iter__(self) -> LatticeDictIterator[T]:
        self.reset()
        return self

    def __next__(self) -> tuple[LatticeKey, T] | LatticeKey | T:
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
            if value.is_integer():
                self._level = int(value)
            else:
                raise ValueError(
                    f"Level must be int or float with no fractional part, got {value}"
                )

    @classmethod
    def from_tuple(cls, key: tuple[float, int], name: str | None = None) -> LatticeKey:
        return cls(coord=key[0], level=key[1], name=name)

    def to_tuple(self) -> tuple[float, int, str] | tuple[float, int]:
        if self.name:
            return self.coord, self.level, self.name
        else:
            return self.coord, self.level

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

    def get_lower_level_left(self, level_difference: int = 1) -> LatticeKey:
        assert self.level != 0, "level is 0, no lower level existing"
        return LatticeKey(
            self.coord - 0.5 * level_difference, self.level - level_difference
        )

    def get_lower_level_right(self, level_difference: int = 1) -> LatticeKey:
        assert self.level != 0, "level is 0, no lower level existing"
        return LatticeKey(
            self.coord + 0.5 * level_difference, self.level - level_difference
        )

    def get_higher_level_right(self, level_difference: int = 1) -> LatticeKey:
        return LatticeKey(
            self.coord + 0.5 * level_difference, self.level + level_difference
        )

    def get_higher_level_left(self, level_difference: int = 1) -> LatticeKey:
        return LatticeKey(
            self.coord - 0.5 * level_difference, self.level + level_difference
        )

    def shift_coord(self, n: int) -> LatticeKey:
        return LatticeKey(coord=self.coord + n, level=self.level, name=self.name)

    def right_up(self, level_difference: int):
        """
        Computes LatticeKey which has level_difference higher level and level_difference/2 larger coord
        i.e. it interprets self as the left lower corner of a Triangle of height level_difference and returns
        the top value.
        """
        return LatticeKey(
            self.coord + 0.5 * level_difference, self.level + level_difference
        )

    def left_up(self, level_difference: int):
        """Same as right_up but with opposite shift in coord."""
        return LatticeKey(
            self.coord - 0.5 * level_difference, self.level + level_difference
        )


def keys_from_iterable(keys: Iterable[tuple[float, int]]) -> list[LatticeKey]:
    return list(map(lambda x: LatticeKey.from_tuple(x), keys))
