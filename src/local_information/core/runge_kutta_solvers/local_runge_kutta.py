from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from local_information.config import RungeKuttaConfig
from local_information.core.runge_kutta_solvers.runge_kutta_solver import (
    RungeKuttaSolver,
)
from local_information.core.utils import anti_commutator

if TYPE_CHECKING:
    from local_information.operators.lindbladian import Lindbladian
    from local_information.operators.hamiltonian import Hamiltonian

logger = logging.getLogger()


class LocalRungeKuttaSolver(RungeKuttaSolver):
    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        hamiltonian: Hamiltonian,
    ):
        super().__init__(
            runge_kutta_config=runge_kutta_config,
            range_=range_,
            system_operator=hamiltonian,
        )

    def dissipator(
        self, key: tuple[float, int], density_matrix: np.ndarray
    ) -> np.ndarray | None:
        # no dissipators present in Hamiltonian evolution
        return None

    @property
    def hamiltonian(self):
        return self._system_operator


class LocalLindbladRungeKuttaSolver(RungeKuttaSolver):
    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        lindbladian: Lindbladian,
    ):
        super().__init__(
            runge_kutta_config=runge_kutta_config,
            range_=range_,
            system_operator=lindbladian,
        )

    def dissipator(
        self, key: tuple[float, int], density_matrix: np.ndarray
    ) -> np.ndarray | None:
        """! Computes the dissipator of the Lindblad equation for the Lindblad operator L"""

        lindbladian_dict_entry = self._system_operator.lindbladian_dict[key]
        ell = int(key[1])
        D = np.zeros((2 ** (ell + 1), 2 ** (ell + 1)), dtype=np.complex128)

        count_non_zero_L = 0
        for e, dict_entry in enumerate(lindbladian_dict_entry):
            # dict_entry is either None or list; if list then it has the form [(type,coupling),()...]
            if dict_entry is None:
                continue
            else:
                count_non_zero_L += 1
                for entry in dict_entry:
                    tpe = entry[0]
                    coupling = entry[1]
                    id_ = (ell, e, tpe)
                    L = self._system_operator.L_operators[id_].toarray()
                    # L_operators is a LatticeDict with keys (ell,m,tpe)
                    L_dagger = np.conjugate(np.transpose(L))
                    D += coupling * (
                        L @ density_matrix @ L_dagger
                        - 0.5 * anti_commutator(L_dagger @ L, density_matrix)
                    )

        if count_non_zero_L == 0:
            return None
        else:
            return D

    @property
    def lindbladian(self):
        return self._system_operator
