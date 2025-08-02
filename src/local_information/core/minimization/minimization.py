from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from local_information.config import TimeEvolutionConfig
from local_information.core.minimization.conjugate_gradient import (
    ConjugateGradientOptimizer,
)
from local_information.core.minimization.projectors import Projector
from local_information.core.utils import compute_lower_level
from local_information.core.utils import compute_mutual_information_at_level
from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi import MultiProcessing
from local_information.mpi.mpi_funcs import get_mpi_variables

if TYPE_CHECKING:
    from local_information.typedefs import SystemOperator, LatticeDictKeyTuple
    from local_information.state.state import State
logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class InformationMinimizer:
    initial_total_information: float

    def __init__(self, config: TimeEvolutionConfig, system_operator: SystemOperator):
        self._config = config
        self._system_operator = system_operator

    def __call__(self, state: State, checks: bool = False) -> LatticeDict:
        self.initial_total_information = (
            self._reduce_and_compute_total_mutual_information(state=state)
        )
        density_matrices = self._minimize_with_fixed_information_current(
            state.density_matrix,
            checks=checks,
        )
        return density_matrices

    def _reduce_and_compute_total_mutual_information(self, state: State) -> float:
        state.reduce_to_level(self._config.min_l)
        total_information = total_mutual_information_at_level(
            self._config.min_l, state.density_matrix
        )
        total_information = COMM.bcast(total_information, root=0)
        return total_information

    def _minimize_with_fixed_information_current(
        self,
        rho_dict: LatticeDict,
        checks=True,
    ) -> LatticeDict:
        """
        Minimizes the information of rho_dict on level min_l specified in config.
        Convergence is assumed when the relative information difference drops below
        the value specified in 'config.minimization_tolerance'.
        """
        convergence_reached = False
        count_minimizer_steps = 0
        total_information = self.initial_total_information
        rho_dict += build_lower_level(rho_dict, self._config.min_l)

        while not convergence_reached:
            # n_max + 1 is necessary here to include the last site
            solution = self._conjugate_gradient(
                rho_dict,
                self._config.min_l,
                checks=checks,
            )

            if RANK == 0:
                rho_dict = (
                    rho_dict
                    + (
                        1.0
                        - self._config.minimization_config.conjugate_gradient_damping
                    )
                    * solution
                )

            (
                convergence_reached,
                total_information,
            ) = self._is_converged_with_new_total_information(
                rho_dict=rho_dict, total_information=total_information
            )
            count_minimizer_steps += 1

            convergence_reached = COMM.bcast(convergence_reached, root=0)

        if RANK == 0:
            logger.info(
                f"Minimization steps to reach convergence: {count_minimizer_steps}"
            )
            if checks:
                self._check_density_matrix(self._config.min_l, rho_dict, tol=1e-14)

        return rho_dict

    @MultiProcessing(shift=0, method=True)
    def _conjugate_gradient(
        self,
        rho_dict: LatticeDict,
        level: int,
        checks: bool = False,
    ) -> LatticeDict:
        if SIZE != 1:
            # builds the lower level for all worker processes
            rho_dict += build_lower_level(rho_dict, level)

        eig_dict = diagonalize_rho_dict(level, rho_dict)
        if checks:
            self._check_eigenvalues(eig_dict)

        projector = Projector(
            system_operator=self._system_operator,
            density_matrix_dict=rho_dict,
            eigen_dict=eig_dict,
        )
        conjugate_gradient_optimizer = ConjugateGradientOptimizer(
            projector=projector, config=self._config
        )
        solution_dict = conjugate_gradient_optimizer.optimize(
            level=level, checks=checks
        )

        return solution_dict

    def _is_converged_with_new_total_information(
        self, rho_dict: LatticeDict, total_information: float
    ) -> tuple[bool, float]:
        new_total_information = total_mutual_information_at_level(
            self._config.min_l, rho_dict
        )
        convergence_reached = (
            (total_information - new_total_information) / self.initial_total_information
            < self._config.minimization_config.minimization_tolerance
        )

        return convergence_reached, new_total_information

    @staticmethod
    def _check_eigenvalues(
        eigen_dict: dict[LatticeDictKeyTuple, tuple[np.ndarray, np.ndarray]],
    ):
        for key, (eigenvalues, _) in eigen_dict.items():
            if any(eigenvalues < 0):
                raise ValueError(
                    f"Unphysical eigenvalues: negative eigenvalues of density matrix at {key}"
                )

    @staticmethod
    def _check_density_matrix(ell, rho_dict, tol=1e-8):
        """
        Check if entries in LatticeDict object are proper density matrices
        """

        for key, rho in rho_dict.items_at_level(ell):
            # check if it's a square matrix
            if not all(len(row) == len(rho) for row in rho):
                raise AssertionError(
                    "input incorrect: density matrix is not a square matrix"
                )

            # check if it has trace of 1
            if not np.isclose(np.trace(rho), 1, atol=tol):
                logger.error("minimized rho has not unit trace")
                logger.info("key", key)
                logger.info(f"np.trace(rho)= {np.trace(rho)}")

                e, v = np.linalg.eigh(rho)
                logger.error(np.max(e))
                raise ValueError(f"density matrix trace deviates from 1 (>{tol})")

            # check if it's semi-positive definite
            e, v = np.linalg.eigh(rho)
            if e.any() <= 0.0:
                raise AssertionError(
                    "input incorrect: density matrix is not semi-positive definite"
                )


def total_mutual_information_at_level(level, rho_dict) -> float:
    _, inf_dict = compute_mutual_information_at_level(rho_dict, level)
    return sum(info for info in inf_dict.values_at_level(level))


def diagonalize_rho_dict(level: int, density_matrix_dict: LatticeDict) -> dict:
    """
    Diagonalizes all density matrices at `level` and
    stores the eigenvalues and eigenvectors in a dict with the same keys
    """

    return {
        key: np.linalg.eigh(density_matrix)
        for key, density_matrix in density_matrix_dict.items_at_level(level)
    }


def build_lower_level(rho_dict: LatticeDict, ell: int) -> LatticeDict:
    """Computes the density matrices one level lower than `ell`"""
    density_matrices_on_lower_level = compute_lower_level(rho_dict, ell)
    return density_matrices_on_lower_level
