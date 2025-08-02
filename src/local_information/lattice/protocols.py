from abc import ABC


class Arithmetics(ABC):
    @classmethod
    def __subclasshook__(cls, C):
        if cls is Arithmetics:
            if (
                any("__add__" in B.__dict__ for B in C.__mro__)
                and any("__sub__" in B.__dict__ for B in C.__mro__)
                and any("__mul__" in B.__dict__ for B in C.__mro__)
            ):
                return True
        return NotImplemented


class Comparable(ABC):
    @classmethod
    def __subclasscheck__(cls, C):
        if cls is Comparable:
            if any("__eq__" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
