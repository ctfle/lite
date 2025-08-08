from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

from local_information.config import RungeKuttaConfig
from local_information.core.petz_map import ptrace
from local_information.core.runge_kutta_solvers.runge_kutta_parameters import (
    RK23_parameters,
    RK45_parameters,
    RK810_parameters,
    RK1012_parameters,
)
from local_information.core.utils import commutator
from local_information.core.utils import get_higher_level
from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables
from local_information.state.state import State
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_information.typedefs import SystemOperator

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class RungeKuttaSolver(ABC):
    """Runge Kutta solver"""

    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        system_operator: SystemOperator,
    ):
        self.config = runge_kutta_config

        if self.config.RK_order == "23":
            self.c, self.a, self.b_higher, self.b_lower = RK23_parameters()
            self.order = 3
        elif self.config.RK_order == "45":
            self.c, self.a, self.b_higher, self.b_lower = RK45_parameters()
            self.order = 5
        elif self.config.RK_order == "1012":
            self.c, self.a, self.b_higher = RK1012_parameters()
            self.b_lower = None
            self.order = 12
        elif self.config.RK_order == "810":
            self.c, self.a, self.b_higher = RK810_parameters()
            self.b_lower = None
            self.order = 8
        else:
            raise ValueError(f"no RK solver for given order: {self.config.RK_order}")

        self._step_size = self.config.step_size
        self._max_error = self.max_error

        self.range_ = range_
        self._system_operator = system_operator

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        """! Setter function for threshold"""
        if value < 0:
            raise ValueError("step_size must be >0")
        self._step_size = value

    @property
    def max_error(self):
        return self.config.max_error

    @max_error.setter
    def max_error(self, value):
        """! Setter function for threshold"""
        if value < 0:
            raise ValueError("max_error must be >0")
        self.config.max_error = value

    @abstractmethod
    def dissipator(
        self, key: tuple[float, int], density_matrix: np.ndarray
    ) -> np.ndarray | None:
        pass

    def solve(
        self, state: State, final_time: float | None = None
    ) -> tuple[LatticeDict, float]:
        """Compute single runge kutta step."""
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

            rhs = self.runge_kutta_func(rho_dict, dyn_max_l)

            k = [self.step_size * rhs]

            for j in range(1, len(self.b_higher)):
                addition = LatticeDict()
                for ell in range(j):
                    addition = addition + self.a[j, ell] * k[ell]
                rhs = self.runge_kutta_func(rho_dict + addition, dyn_max_l)
                k += [self.step_size * rhs]

            # sum up
            if self.order == 12:
                for j, _ in enumerate(self.b_higher):
                    total_high = total_high + self.b_higher[j] * k[j]

                # estimated error in the 10th order RK method according to:
                # Neural, Parallel, and Scientific Computations 20 (2012) 437-458
                error = 49 / 640 * (k[1] - k[23])

            elif self.order == 8:
                for j, _ in enumerate(self.b_higher):
                    total_high = total_high + self.b_higher[j] * k[j]

                # estimated error in the 8th order RK method according to:
                # Neural, Parallel, and Scientific Computations 20 (2012) 437-458
                error = 1 / 360 * (k[1] - k[15])

            else:
                # for all other methods the error is estimated from computing different orders
                for j, _ in enumerate(self.b_higher):
                    total_high = total_high + self.b_higher[j] * k[j]
                    total_low = total_low + self.b_lower[j] * k[j]

                # compute error
                error = total_high - total_low

            # compute allowed stepsize
            allowed_step_size = np.zeros(len(error))
            actual_error = np.linalg.norm(list(error.values()), np.inf, axis=(1, 2))
            logger.debug(f"actual_error is: {actual_error} ")

            for j, _ in enumerate(error):
                if not actual_error[j] < 1e-16:
                    allowed_step_size[j] = (
                        self.step_size
                        * (1 / actual_error[j]) ** (1 / self.order)
                        * self.max_error ** (1 / self.order)
                    )
                else:
                    allowed_step_size[j] = (self.step_size * 1.05) / 0.9

            # compare optimal step size to actual step size
            if np.any(allowed_step_size < self.step_size):
                # redo calculation with smaller stepsize
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

        tot = rho_dict + total_high
        tot_ct = (rho_dict + total_high).dagger()

        logger.info(
            f"finished Runge-Kutta time step with adaptive stepsize {self.step_size}"
        )
        return 0.5 * (tot + tot_ct), time

    def runge_kutta_func(self, rho_dict: LatticeDict, dyn_max_l: int) -> LatticeDict:
        """Computes the right hand side of the von-Neumann equation"""

        sqrt_method = False
        if self.config.petz_map == "sqrt":
            sqrt_method = True
        work_dict = deepcopy(rho_dict)
        # do petz map only if summit of the triangle is not yet reached
        if len(work_dict) != 1:
            # petz_map the density matrices at level max_l +_range
            for r in range(self.range_):
                work_dict += get_higher_level(
                    work_dict, dyn_max_l + r, sqrt_method=sqrt_method
                )
        if work_dict is not None:
            key_max_l_dim = work_dict.dim_at_level(dyn_max_l)
            # compute the rhs of the von-Neumann equation at level max_l and store it in work_dict
            for m, k in enumerate(work_dict.keys_at_level(dyn_max_l)):
                # m is the number of sites left to k
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
