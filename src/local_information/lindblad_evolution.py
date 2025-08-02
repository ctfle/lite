from __future__ import annotations

import logging

from local_information.config import TimeEvolutionConfig
from local_information.config.monitor import DataContainer
from local_information.core.runge_kutta_solvers.local_runge_kutta import (
    LocalLindbladRungeKuttaSolver,
)
from local_information.core.runge_kutta_solvers.remote_runge_kutta import (
    RemoteLindbladRungeKuttaSolver,
)
from local_information.core.minimization.minimization import InformationMinimizer
from local_information.mpi.mpi_funcs import get_mpi_variables
from local_information.operators.lindbladian import Lindbladian
from local_information.state.state import State
from local_information.system import System

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class OpenSystem(System):
    """!
    Class to compute and time evolve subsystem density matrices based on the Petz map and
    the corresponding information lattice and currents. Class objects contain information on the current
    state of the operators and keep all relevant Hamiltonians as well as information on the kind of operators
    (finite, translational invariant, asymptotically invariant)
    """

    def __init__(
        self,
        init_state: State,
        lindbladian: Lindbladian,
        config: TimeEvolutionConfig = TimeEvolutionConfig(),
        data: DataContainer = DataContainer(),
    ):
        super().__init__(
            init_state=init_state, config=config, data=data, operator=lindbladian
        )

        # maximum correlation length. As long as dyn_max_l < max_l, the exact Petz map will be used
        if self.lindbladian.max_l != self.config.max_l:
            logger.info(
                "max_l defined in Hamiltonian does not match max_l in Time_Evolution\n"
            )
            logger.info("Using max_l of Hamiltonian")

        # align hamiltonian and state
        self._align(loaded=self.state.loaded_state)

        if not self.check_compatibility():
            raise AssertionError(
                "State and Lindbladian incompatible: "
                "Hamiltonian not defined on whole system define described by input State"
            )

        # Runge-Kutta solver
        if PARALLEL:
            self.solver = RemoteLindbladRungeKuttaSolver(
                runge_kutta_config=self.config.runge_kutta_config,
                range_=self.range_,
                lindbladian=self.lindbladian,
            )
        else:
            self.solver = LocalLindbladRungeKuttaSolver(
                runge_kutta_config=self.config.runge_kutta_config,
                range_=self.range_,
                lindbladian=self.lindbladian,
            )

    @property
    def lindbladian(self) -> Lindbladian:
        return self._system_operator

    @lindbladian.setter
    def lindbladian(self, lindbladian: Lindbladian):
        """
        Setter for Lindbladian updating also the solver
        """
        # the changed Hamiltonian has to have same length L
        assert lindbladian.L == self.system_size

        self._system_operator = lindbladian
        logger.debug("changed Lindbladian")
        if PARALLEL:
            self.solver = RemoteLindbladRungeKuttaSolver(
                runge_kutta_config=self.config.runge_kutta_config,
                range_=lindbladian.range_,
                lindbladian=lindbladian,
            )
        else:
            self.solver = LocalLindbladRungeKuttaSolver(
                runge_kutta_config=self.config.runge_kutta_config,
                range_=lindbladian.range_,
                lindbladian=lindbladian,
            )
        logger.debug("updated Runge-Kutta solver")
        self.minimize_information = InformationMinimizer(
            config=self.config, system_operator=lindbladian
        )
        logger.debug("updated InformationMinimizer")

    @classmethod
    def from_checkpoint(cls, folder: str, module_path: str):
        config = TimeEvolutionConfig.from_yaml(folder)
        logger.info(f"loaded TimeEvolutionConfig from {folder}")
        state = State.from_checkpoint(folder)
        lindbladian = Lindbladian.from_checkpoint(folder)
        data = DataContainer.from_yaml(folder, module_path)

        return cls(init_state=state, lindbladian=lindbladian, config=config, data=data)

    def __str__(self) -> str:
        if RANK == 0:
            system = "OpenSystem:\n"
            system += super().__str__()
            lindbladian = self.lindbladian.__str__()
            return system + lindbladian
