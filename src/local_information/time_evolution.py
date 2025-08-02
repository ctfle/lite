from __future__ import annotations

import logging

from local_information.config import TimeEvolutionConfig
from local_information.config.monitor import DataContainer
from local_information.core.runge_kutta_solvers.local_runge_kutta import (
    LocalRungeKuttaSolver,
)
from local_information.core.runge_kutta_solvers.remote_runge_kutta import (
    RemoteRungeKuttaSolver,
)
from local_information.core.minimization.minimization import InformationMinimizer
from local_information.mpi.mpi_funcs import get_mpi_variables
from local_information.operators.hamiltonian import Hamiltonian
from local_information.state.state import State
from local_information.system import System

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class ClosedSystem(System):
    """
    Class to compute and time evolve subsystem density matrices based on the Petz map and the corresponding
    information lattice and currents. Class objects contain information on the current state of the operators
    and keep all relevant Hamiltonians as well as information on the kind of operators
    (finite, translational invariant, asymptotically invariant)
    """

    def __init__(
        self,
        init_state: State,
        hamiltonian: Hamiltonian,
        config: TimeEvolutionConfig = TimeEvolutionConfig,
        data: DataContainer = DataContainer,
    ):
        super().__init__(
            init_state=init_state, config=config, data=data, operator=hamiltonian
        )

        # maximum correlation length. As long as dyn_max_l < max_l, the exact Petz map will be used
        if self.max_l != self.config.max_l:
            logger.info(
                "max_l defined in Hamiltonian does not match max_l in Time_Evolution\n"
            )
            logger.info("Using max_l of Hamiltonian")

        # align hamiltonian and state
        self._align(loaded=self.state.loaded_state)

        if not self.check_compatibility():
            raise AssertionError(
                "State and Hamiltonian incompatible: "
                "Hamiltonian not defined on whole system described by input State"
            )

        # Runge-Kutta solver
        if PARALLEL:
            self.solver = RemoteRungeKuttaSolver(
                runge_kutta_config=self.config.runge_kutta_config,
                range_=self.range_,
                hamiltonian=self.hamiltonian,
            )
        else:
            self.solver = LocalRungeKuttaSolver(
                runge_kutta_config=self.config.runge_kutta_config,
                range_=self.range_,
                hamiltonian=self.hamiltonian,
            )

    @property
    def hamiltonian(self) -> Hamiltonian:
        return self._system_operator

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian: Hamiltonian):
        """
        Setter for hamiltonian updating also the solver
        """
        # the changed Hamiltonian has to have same length L
        if hamiltonian.L != self.system_size:
            raise ValueError(
                f"can only update the Hamiltonian on the same system "
                f"with size L={self.system_size}"
            )

        self._system_operator = hamiltonian
        logger.debug("changed Hamiltonian")
        # Runge-Kutta solver
        if PARALLEL:
            self.solver = RemoteRungeKuttaSolver(
                runge_kutta_config=self.config.runge_kutta_config,
                range_=hamiltonian.range_,
                hamiltonian=hamiltonian,
            )
        else:
            self.solver = LocalRungeKuttaSolver(
                runge_kutta_config=self.config.runge_kutta_config,
                range_=hamiltonian.range_,
                hamiltonian=hamiltonian,
            )
        logger.debug("updated Runge-Kutta solver")
        self.minimize_information = InformationMinimizer(
            config=self.config, system_operator=hamiltonian
        )
        logger.debug("updated InformationMinimizer")

    @classmethod
    def from_checkpoint(cls, folder: str, module_path: str) -> ClosedSystem:
        """
        load from checkpoint
        """
        config = TimeEvolutionConfig.from_yaml(folder)
        state = State.from_checkpoint(folder)
        hamiltonian = Hamiltonian.from_checkpoint(folder)
        data = DataContainer.from_yaml(folder, module_path)

        logger.info(f"loaded ClosedSystem from {folder}")
        return cls(init_state=state, hamiltonian=hamiltonian, config=config, data=data)

    def __str__(self) -> str | None:
        if RANK == 0:
            system = "ClosedSystem:\n"
            system += super().__str__()
            hamiltonian = self.hamiltonian.__str__()
            return system + hamiltonian
