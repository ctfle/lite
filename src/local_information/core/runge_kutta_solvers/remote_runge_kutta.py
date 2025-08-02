from __future__ import annotations

import logging
import sys
from copy import deepcopy
import numpy as np

logger = logging.getLogger()
from local_information.mpi.mpi import *
from local_information.mpi.mpi_funcs import get_mpi_variables
from local_information.core.utils import commutator
from local_information.core.utils import get_higher_level
from local_information.core.petz_map import ptrace
from local_information.config import RungeKuttaConfig
from local_information.state.state import State
from local_information.core.runge_kutta_solvers.runge_kutta_solver import (
    RungeKuttaSolver,
)
from local_information.core.utils import anti_commutator
from local_information.typedefs import SystemOperator

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class RemoteRungeKuttaSolver(RungeKuttaSolver):
    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        hamiltonian: SystemOperator,
    ):
        super().__init__(
            runge_kutta_config=runge_kutta_config,
            range_=range_,
            system_operator=hamiltonian,
        )

    def dissipator(
        self, key: tuple[float, int], density_matrix: np.ndarray
    ) -> np.ndarray | None:
        # no dissipator for Hamiltonian evolution
        pass

    @property
    def hamiltonian(self):
        return self._system_operator

    def solve(
        self, state: State, final_time: float | None = None
    ) -> tuple[LatticeDict, float]:
        """! Compute one runge kutta step"""
        rho_dict, dyn_max_l, current_time = (
            state.density_matrix,
            state.dyn_max_l,
            state.current_time,
        )

        repeat = True
        while repeat:
            total_high = LatticeDict()
            total_low = LatticeDict()
            time = None
            k = None

            rhs = self.runge_kutta_func(rho_dict, dyn_max_l)

            if RANK != 0:
                if rhs is not None:
                    logger.debug(f"RANK {RANK} has {rhs}")
                    sys.stdout.flush()
                    raise ValueError

            if RANK == 0:
                k = [self.step_size * rhs]

            for j in range(1, len(self.b_higher)):
                addition = LatticeDict()
                for ell in range(j):
                    if RANK == 0:
                        addition = addition + self.a[j, ell] * k[ell]
                rhs = self.runge_kutta_func(rho_dict + addition, dyn_max_l)
                if RANK == 0:
                    k += [self.step_size * rhs]

            # sum up
            if RANK == 0:
                if self.order == 12:
                    for j, _ in enumerate(self.b_higher):
                        total_high = total_high + self.b_higher[j] * k[j]

                    # estimated error in the 10th order RK method according to:
                    # Neural, Parallel, and Scientific Computations 20 (2012) 437-458
                    error = 49 / 640 * (k[1] - k[23])

                elif self.order == 8:
                    for j, _ in enumerate(self.b_higher):
                        total_high = total_high + self.b_higher[j] * k[j]

                    # estimated error in the 10th order RK method according to:
                    # Neural, Parallel, and Scientific Computations 20 (2012) 437-458
                    error = 1 / 360 * (k[1] - k[15])

                else:
                    for j, _ in enumerate(self.b_higher):
                        total_high = total_high + self.b_higher[j] * k[j]
                        total_low = total_low + self.b_lower[j] * k[j]

                    error = total_high - total_low

                # compute error
                allowed_step_size = np.zeros(len(error))
                actual_error = np.linalg.norm(list(error.values()), np.inf, axis=(1, 2))

                for j, _ in enumerate(error):
                    if not actual_error[j] < 1e-16:
                        allowed_step_size[j] = self.step_size * (
                            self.max_error / actual_error[j]
                        ) ** (1 / self.order)
                    else:
                        allowed_step_size[j] = (self.step_size * 1.05) / 0.9

                # compare optimal step size to actual step size
                if np.any(allowed_step_size < self.step_size):
                    # redo calculation with smaller step size
                    self.step_size = np.min(allowed_step_size) * 0.9
                else:
                    # repeat to land exactly on the final time
                    if (
                        final_time is not None
                        and current_time + self.step_size > final_time
                    ):
                        self.step_size = final_time - current_time
                    else:
                        time = current_time + self.step_size
                        # update the step size
                        self.step_size = np.min(allowed_step_size) * 0.9
                        repeat = False

            repeat = COMM.bcast(repeat, root=0)

        time = COMM.bcast(time, root=0)
        self.step_size = COMM.bcast(self.step_size, root=0)

        tot = rho_dict + total_high
        tot_ct = (rho_dict + total_high).dagger()

        logger.info(
            f"finished Runge-Kutta time step with adaptive stepsize {self.step_size}"
        )
        return 0.5 * (tot + tot_ct), time

    def runge_kutta_func(self, rho_dict: LatticeDict, dyn_max_l: int):
        """Computes the right hand side of the von-Neumann equation"""

        sqrt_method = False
        if self.config.petz_map == "sqrt":
            sqrt_method = True
        work_dict = deepcopy(rho_dict)
        # do petz map only if summit of the triangle is not yet reached
        if len(work_dict) != 1:
            # petz_map the density matrices at level max_l +_range
            for r in range(self.range_):
                higher_level_dict = get_higher_level(
                    work_dict, dyn_max_l + r, sqrt_method=sqrt_method
                )
                if RANK == 0:
                    work_dict += higher_level_dict
                else:
                    work_dict = None
            # necessary for the case where range_ = 0
            if not RANK == 0:
                work_dict = None

        if work_dict is not None:
            key_max_l_dim = work_dict.dim_at_level(dyn_max_l)
            # compute the rhs of the von-Neumann equation at level max_l and store it in work_dict
            for m, k in enumerate(work_dict.keys_at_level(dyn_max_l)):
                # m_ is the number of sites left to k
                if m >= self.range_:
                    # the site is at least _range away form the boundary
                    key_l = (k[0] - 0.5 * self.range_, dyn_max_l + self.range_)
                    DM_l = work_dict[key_l]
                    H_max_l_range = self._system_operator.subsystem_hamiltonian[key_l]
                    # build commutator and trace out _range sites on the left
                    _com_l = ptrace(
                        commutator(H_max_l_range.toarray(), DM_l),
                        self.range_,
                        end="left",
                    )

                else:
                    # the site is less than _range away form the left boundary
                    key_l = (k[0] - 0.5 * m, dyn_max_l + m)
                    DM_l = work_dict[key_l]
                    H_max_l_m_ = self._system_operator.subsystem_hamiltonian[key_l]
                    # build commutator and trace out _m_ sites on the left
                    _com_l = ptrace(
                        commutator(H_max_l_m_.toarray(), DM_l), m, end="left"
                    )

                # repeat the same for the right side
                bar_m = (key_max_l_dim - 1) - m
                if bar_m >= self.range_:
                    # the site is at least _range away form the boundary
                    key_r = (k[0] + 0.5 * self.range_, dyn_max_l + self.range_)
                    DM_r = work_dict[key_r]
                    H_max_l_range = self._system_operator.subsystem_hamiltonian[key_r]
                    # build commutator and trace out _range sites on the left
                    _com_r = ptrace(
                        commutator(H_max_l_range.toarray(), DM_r),
                        self.range_,
                        end="right",
                    )

                else:
                    # the site is at less than _range away form the right boundary
                    key_r = (k[0] + 0.5 * bar_m, dyn_max_l + bar_m)
                    DM_r = work_dict[key_r]
                    H_max_l_m_ = self._system_operator.subsystem_hamiltonian[key_r]
                    # build commutator and trace out _range sites on the left
                    _com_r = ptrace(
                        commutator(H_max_l_m_.toarray(), DM_r), bar_m, end="right"
                    )

                # the term at max_l is always the same
                DM_c = work_dict[k]
                H_max_l = self._system_operator.subsystem_hamiltonian[k]
                _com_c = commutator(H_max_l.toarray(), DM_c)
                rhs = _com_l + _com_r - _com_c

                # Lindblad terms: in this implementation only onsite terms are allowed
                D = self.dissipator(k, DM_c)
                if D is not None:
                    rhs += 1j * D

                work_dict[k] = -1j * rhs

            # drop everything not at level max_l
            for key in list(work_dict):
                if key[1] != dyn_max_l:
                    work_dict.pop(key, None)

        return work_dict


class RemoteLindbladRungeKuttaSolver(RemoteRungeKuttaSolver):
    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        lindbladian: SystemOperator,
    ):
        super().__init__(
            runge_kutta_config=runge_kutta_config,
            range_=range_,
            hamiltonian=lindbladian,
        )

    def dissipator(
        self, key: tuple[float, int], density_matrix: np.ndarray
    ) -> np.ndarray | None:
        """!
        Computes the dissipator of the Lindblad equation for the Lindblad operator L
        """

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
