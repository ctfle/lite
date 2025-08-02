from __future__ import annotations

from typing import Union, TypeVar

from local_information.core.runge_kutta_solvers.runge_kutta_solver import (
    RungeKuttaSolver,
)
from local_information.operators.hamiltonian import Hamiltonian
from local_information.operators.lindbladian import Lindbladian

Coupling = TypeVar(
    "Coupling",
    list[list[str, list]],
    list[list[str, float]],
    list[list[Union[str, list[float]]]],
    list[list[bool], list[list[Union[str, list[float]]]]],
)
Solver = RungeKuttaSolver
SystemOperator = Union[Hamiltonian, Lindbladian]
LatticeDictKeyTuple = tuple[float, int]
