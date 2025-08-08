from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from numbers import Number
from typing import TYPE_CHECKING

from mpi4py import MPI

from local_information.config import TimeEvolutionConfig
from local_information.config.monitor import DataContainer
from local_information.core.minimization.minimization import InformationMinimizer
from local_information.core.utils import push_keys, one_shift, Trigger, Status
from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables
from local_information.state.state import State

if TYPE_CHECKING:
    from local_information.typedefs import Solver, SystemOperator

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class System(ABC):
    solver: Solver

    def __init__(
        self,
        init_state: State,
        operator: SystemOperator,
        config: TimeEvolutionConfig,
        data: DataContainer,
    ):
        # initial state
        self.state = init_state
        # config dataclass
        self.config = config
        # TimeEvolutionData
        self.data = data
        # Operator for time-evolution
        self._system_operator = operator
        # Minimizer performing the constraint conjugate gradient optimization of information
        self.minimize_information = InformationMinimizer(
            config=self.config, system_operator=self._system_operator
        )
        # Solver to solve the time-evolution ODE
        self.solver: Solver

        # helper field used to track the time evolution process
        self._trigger = Trigger()

    @property
    def max_l(self):
        return self._system_operator.max_l

    @property
    def range_(self):
        return self._system_operator.range_

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, folder: str, module_path: str):
        pass

    @property
    def dyn_max_l(self):
        return self.state.dyn_max_l

    @property
    def system_size(self):
        return self._system_operator.L

    def save_checkpoint(self):
        self.data.save_checkpoint(self.config.checkpoint_folder)
        self.state.save_checkpoint(self.config.checkpoint_folder)
        self._system_operator.save_checkpoint(self.config.checkpoint_folder)
        self.config.to_yaml()
        pass

    def attach_to_existing_file(self):
        self.data.attach_to_existing_file(self.config.checkpoint_folder)
        self.state.save_checkpoint(self.config.checkpoint_folder)
        self._system_operator.save_checkpoint(self.config.checkpoint_folder)
        self.config.to_yaml()
        pass

    def _shift(self) -> LatticeDict:
        """!
        shifts the density matrices and returns shift dict for un-shifting
        """
        s = None
        if self.config.shift > 0:
            s = one_shift(self.state.density_matrix, self.config.shift)
            self.state.density_matrix = (
                1 / (1 + self.config.shift) * (self.state.density_matrix + s)
            )
        return s

    def _unshift(self, s: LatticeDict):
        if self.config.shift > 0:
            self.state.density_matrix = (
                self.state.density_matrix * (1 + self.config.shift) - s
            )
        pass

    def _align(self, loaded: bool = False):
        """
        Align hamiltonian and state.
        """
        # set the center of the operators i.e. initial state at the center of the defined Hamiltonians
        if self._system_operator.L % 2 == 0:
            anchor = self.system_size // 2
        else:
            anchor = self.system_size // 2 + 1

        if self.state.case == "finite":
            if self.state.density_matrix.dim_at_level(
                self.dyn_max_l
            ) != self._system_operator.subsystem_hamiltonian.dim_at_level(
                self.dyn_max_l
            ):
                raise ValueError("state must be defined on the entire system")

        # shift everything by +L/2 -ell/2 + len(keys)//2 to fix the center of the initial state to
        # the center of all pre-initialized Hamiltonians
        # NOTE: the 0.5 is necessary so that the lowest level has integer numbering.
        # Otherwise, one obtains key errors form subsystem_hamiltonian
        if not loaded:
            if self.dyn_max_l % 2 == 0:
                self.state.density_matrix = push_keys(
                    self.state.density_matrix,
                    anchor
                    - len(self.state.density_matrix.keys()) // 2
                    - self.dyn_max_l / 2,
                )
                # shift state anchor accordingly
                self.state.anchor += (
                    anchor
                    - len(self.state.density_matrix.keys()) // 2
                    - self.dyn_max_l / 2
                )
            else:
                self.state.density_matrix = push_keys(
                    self.state.density_matrix,
                    anchor
                    - len(self.state.density_matrix.keys()) // 2
                    - self.dyn_max_l / 2
                    - 0.5,
                )
                self.state.anchor += (
                    anchor
                    - len(self.state.density_matrix.keys()) // 2
                    - self.dyn_max_l / 2
                    - 0.5
                )
        else:
            # state is loaded from previous initialization -- no shift required
            logger.info("use loaded state")
        pass

    def _update_dyn_max_l(self):
        """
        Updates the dynamical length dyn_max_l if information
        has spread to larger scales in the finite operators case.
        """

        # exit if one process has reached the top
        # of the info-pyramid, i.e. if RANK==0 has reached the summit
        val = len(self.state.density_matrix.keys())
        minVal = COMM.allreduce(val, op=MPI.MIN)
        if minVal == 1:
            return

        # compute information on level dyn_max_l
        if self._trigger.status is Status.Continue:
            self.state.update_dyn_max_l(
                threshold=self.config.update_dyn_max_l_threshold,
                nr_of_updates=self.range_,
            )

    def measure(self, return_values=False) -> dict[str, Number] | None:
        """
        Measure the observables and store results in data
        :param bool return_values: If True, measured observables are returned as dict.
                                    Default False
        """

        (
            information_dict,
            rho_dict_all_levels,
        ) = self.state.get_information_lattice()
        self.state.current_sum_info = sum(
            information_dict.values_at_level(self.dyn_max_l)
        )

        # self.state.total_information = np.sum(list(information_dict.values()))
        self.data.update_default_observables(
            density_matrix=rho_dict_all_levels,
            information_dict=information_dict,
            state=self.state,
            operator=self._system_operator,
        )
        self.data.update_custom_observables(rho_dict_all_levels)

        del rho_dict_all_levels
        self.state.current_sum_info = COMM.bcast(self.state.current_sum_info, root=0)
        if return_values:
            return self.data.return_latest()

    def evolve(
        self, max_evolution_time: float, final_time=False, checks=False
    ) -> DataContainer:
        """
        Computes time evolution of the information lattice density matrices for max_evolution_time
        :param float max_evolution_time: evolution time to evolve for
        :param bool final_time: evolve exactly to the final time or not.
                If True, final_time = starting_time + max_evolution_time. Default False
        :param bool checks: Checks the eigenvalues of the density matrices when minimizing the information
        :returns: TimeEvolutionData containing the measured observables
        :rtype: TimeEvolutionData
        """
        self.state.current_time = self.state.starting_time
        if final_time:
            final_time = self.state.current_time + max_evolution_time
        else:
            final_time = None

        self._set_trigger()

        while self.state.current_time < self.state.starting_time + max_evolution_time:
            self._time_evolution_unit(final_time=final_time, checks=checks)
            # Remove this line
            logger.info(
                f"system size: {self.state.system_size}, total time:  {self.state.current_time}"
            )
            # check if the operators has only a single density matrix: the top is reached
            if len(list(self.state.density_matrix.keys())) == 1:
                self._enlarge_state(additional_sites=1)

            self.log_status(log_level=logging.DEBUG)
            COMM.Barrier()

        # update parameters before saving
        self.config.runge_kutta_config.step_size = float(self.solver.step_size)
        self.state.starting_time = self.state.current_time
        if self.config.save_checkpoint:
            self.save_checkpoint()
            logger.info(f"saved checkpoint under {self.config.checkpoint_folder}")

        return self.data

    def _set_trigger(self):
        if self.dyn_max_l + self.range_ > self.max_l:
            self._trigger.pull()
        else:
            self._trigger.reset()
        self._trigger = COMM.bcast(self._trigger, root=0)

    def _time_evolution_unit(self, final_time: float | None, checks: bool):
        """
        A single evolution cycle including 1) a time evolution step 2) measuring observables
        3) minimization of information (if triggered) 4) level update (if triggered)
        """
        # perform one Runge-Kutta step
        self._time_evolution_step(final_time=final_time)
        # measure observables
        self.measure(return_values=False)
        # minimize information if required
        self._minimize(checks=checks)

        # update level if required
        self._update_dyn_max_l()
        self._set_trigger()

    def _minimize(self, checks: bool):
        """
        Minimize information at level min_l if enough information accumulated at 'max_l'.
        """
        # Remove this line
        logger.info(
            f"total information at dyn_max_l: {self.state.total_information_at_dyn_max_l} "
        )
        total_information_at_dyn_max_l = self.state.total_information_at_dyn_max_l
        total_information_at_dyn_max_l = COMM.bcast(
            total_information_at_dyn_max_l, root=0
        )
        total_information = self.state.total_information
        total_information = COMM.bcast(total_information, root=0)
        if (
            total_information_at_dyn_max_l
            > self.config.minimization_config.minimization_threshold * total_information
            and self._trigger.status is Status.Stop
        ):
            logger.info(
                "Minimize information at level {0:d} at time = {1:0.3f}".format(
                    self.config.min_l, self.state.current_time
                )
            )
            self.state.density_matrix = self.minimize_information(
                state=self.state, checks=checks
            )

            if RANK == 0:
                self.state.density_matrix.kill_all_except(self.config.min_l)
            if not self.max_l == self.config.min_l:
                self._trigger.reset()

    def _time_evolution_step(self, final_time: float | None):
        """
        Single time evolution step:
        Shift the density matrices for numerical stability,
        update the max acceptable error,
        use the solver to solve the Schrödinger equation for a small adaptive time step,
        unshift the evolved density matrices.
        """
        # add _range +1 sites at each end
        additional_sites_l, additional_sites_r = self._enlarge_state(
            additional_sites=self.range_ + 2
        )

        # one-shift for numerical stability
        shift_dict = self._shift()
        # rescale max_error for RK solver
        tmp_max_err = self.config.runge_kutta_config.max_error
        self.solver.max_error = (
            1 + self.config.shift
        ) * self.config.runge_kutta_config.max_error
        # evolve the enlarged operators with the finite mechanism
        self.state.density_matrix, self.state.current_time = self.solver.solve(
            self.state, final_time=final_time
        )
        # undo one-shift
        self._unshift(s=shift_dict)
        self.solver.max_error = tmp_max_err
        # check the convergence and remove unnecessary sites
        self.state.check_convergence(
            sites_to_check_left=additional_sites_l,
            sites_to_check_right=additional_sites_r,
            tolerance=self.config.system_size_tol,
        )

    def _compute_additional_sites(self, additional_sites: int) -> tuple[int, int]:
        """
        Compute the number of sites that are to be attached to the state.
        """
        # check if system is defined including additional sites
        (
            operator_min_n,
            operator_max_n,
        ) = self._system_operator.subsystem_hamiltonian.boundaries(self.dyn_max_l)
        state_min_n, state_max_n = self.state.density_matrix.boundaries(self.dyn_max_l)

        # compare the boundaries right
        if operator_max_n - state_max_n > additional_sites:
            additional_sites_r = additional_sites
        else:
            if operator_max_n - state_max_n > 0:
                additional_sites_r = int(operator_max_n - state_max_n)
            else:
                additional_sites_r = 0
                logger.debug("state reached the boundary on the right")

        # compare the boundaries left
        if abs(operator_min_n - state_min_n) > additional_sites:
            additional_sites_l = additional_sites
        else:
            if abs(operator_min_n - state_min_n) > 0:
                additional_sites_l = int(abs(operator_min_n - state_min_n))
            else:
                additional_sites_l = 0
                logger.debug("state reached the boundary on the left")

        return additional_sites_l, additional_sites_r

    def _enlarge_state(self, additional_sites: int) -> tuple[int, int]:
        """
        Enlarge the state with `additional_sites` on the right and left end.
        This is only possible if the Hamiltonian is defined on those sites. Thus, the actual
        number of sites added can deviate from the input and is returned.
        In case we deal with a finite system, the state is defined on the entire realm of the
        Hamiltonian and no sites get added."""
        nr_of_sites_left, nr_of_sites_right = self._compute_additional_sites(
            additional_sites
        )
        self.state.enlarge_right(nr_of_sites_right)
        self.state.enlarge_left(nr_of_sites_left)
        return nr_of_sites_left, nr_of_sites_right

    def __str__(self) -> str:
        info = "TimeEvolutionConfig:\n"
        info += self.config.__str__()
        status_info = self.log_status()
        info += status_info

        return info

    def log_status(self, log_level: int = logging.INFO) -> str:
        status_info = ""
        status_info += "\n\t{:<25}: {}\n".format("system size", self.state.system_size)
        status_info += "\t{:<25}: {}\n".format("dyn_max_l", self.dyn_max_l)
        status_info += "\t{:<25}: {}\n".format(
            "total information at dyn_max_l",
            self.state.total_information_at_dyn_max_l,
        )
        status_info += "\t{:<25}: {}\n".format("time", self.state.current_time)
        logger.log(log_level, status_info)
        return status_info

    def check_compatibility(self):
        """
        Check that the Hamiltonian is at least defined over the whole operator
        """
        ell = self.dyn_max_l
        assert ell == list(self.state.density_matrix.keys())[0][1]
        n_min = self.state.density_matrix.smallest_at_level(ell)
        n_max = self.state.density_matrix.largest_at_level(ell)
        N = n_max - n_min

        return N + ell < self.system_size
