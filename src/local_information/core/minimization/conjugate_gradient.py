from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from local_information.config.config import TimeEvolutionConfig
from local_information.core.minimization.projectors import Projector
from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables

if TYPE_CHECKING:
    pass
logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class ConjugateGradientOptimizer:
    def __init__(self, projector: Projector, config: TimeEvolutionConfig):
        self._projector = projector
        self._config = config

    def optimize(self, level: int, checks: bool = False) -> LatticeDict:
        """
        Inner preconditioned conjugate gradient routine:
        https://en.wikipedia.org/wiki/Conjugate_gradient_method
        """
        solution_dict = LatticeDict()
        for key in self._projector.density_matrices.keys_at_level(level):
            xk = np.zeros(
                (2 ** (level + 1), 2 ** (level + 1)),
                np.float64,
            )
            b = self._projector.project_gradient_to_fixed_current(key)
            rk = b - self._projector.project_hessian_to_fixed_current(key, xk)
            zk = self._projector.precondition_to_fixed_current(key, rk)
            pk = zk

            rk_norm = np.linalg.norm(rk, np.inf)
            num_iter = 0
            while (
                rk_norm > self._config.minimization_config.conjugate_gradient_tolerance
            ):
                apk = self._projector.project_hessian_to_fixed_current(key, pk)
                rkzk = inner_product(rk, zk)
                alpha = rkzk / inner_product(pk, apk)
                xk = xk + alpha * pk
                rk = rk - alpha * apk

                rk_norm = np.linalg.norm(rk, np.inf)
                if (
                    rk_norm
                    < self._config.minimization_config.conjugate_gradient_tolerance
                ):
                    continue
                else:
                    zk = self._projector.precondition_to_fixed_current(key, rk)
                    beta = inner_product(rk, zk) / rkzk
                    pk = zk + beta * pk
                    num_iter += 1

            solution_dict[key] = xk

            if checks:
                if np.linalg.norm(np.trace(solution_dict[key]) > 10 ** (-16)):
                    logger.info(
                        "np.trace(solution_dict[key])", np.trace(solution_dict[key])
                    )
        return solution_dict


def inner_product(A: np.ndarray, B: np.ndarray) -> float:
    """
    Inner product for two matrices A and B, defined
    as the sum of all element-wise multiplied elements
    """
    return np.sum(np.conjugate(A) * B)
