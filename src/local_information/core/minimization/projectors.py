from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from local_information.core.petz_map import ptrace
from local_information.core.utils import arctanh, commutator, np_logm
from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables

if TYPE_CHECKING:
    from local_information.typedefs import SystemOperator, LatticeDictKeyTuple
logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class Projector:
    def __init__(
        self,
        system_operator: SystemOperator,
        density_matrix_dict: LatticeDict,
        eigen_dict: dict[LatticeDictKeyTuple, tuple[np.ndarray, np.ndarray]],
    ):
        """
        eigen_dict here contains eigenvalues and eigenvectors associated with the density matrices
        provided by density_matrix_dict
        """
        self._system_operator = system_operator
        self._density_matrix_dict = density_matrix_dict
        self._eigen_dict = eigen_dict

    @property
    def density_matrices(self):
        return self._density_matrix_dict

    @property
    def range_(self) -> int:
        return self._system_operator.range_

    def _eigen_vectors(self, key: tuple[float, int]) -> np.ndarray:
        return self._eigen_dict[key][1]

    def _eigen_values(self, key: tuple[float, int]) -> np.ndarray:
        return self._eigen_dict[key][0]

    def _get_hessian_matrix_elements(self, key: tuple[float, int]) -> np.ndarray:
        """
        Computes the elements h_ij that define the Hessian operator H; they're just a function of
        the eigenvalues of rho
        """
        density_matrix_eigenvalues = self._eigen_values(key)
        length = len(density_matrix_eigenvalues)
        matrix = np.zeros((length, length), dtype=np.float64)

        for j in range(length):
            for k in range(j + 1):
                if density_matrix_eigenvalues[j] != density_matrix_eigenvalues[k]:
                    matrix[j, k] = -arctanh(
                        (density_matrix_eigenvalues[j] - density_matrix_eigenvalues[k])
                        / (
                            density_matrix_eigenvalues[j]
                            + density_matrix_eigenvalues[k]
                        )
                    ) / (density_matrix_eigenvalues[j] - density_matrix_eigenvalues[k])
                else:
                    matrix[j, k] = -1 / (
                        density_matrix_eigenvalues[j] + density_matrix_eigenvalues[k]
                    )

        # symmetrize
        matrix = matrix + np.conjugate(np.transpose(matrix - np.diag(np.diag(matrix))))

        return matrix

    def hessian(self, key: LatticeDictKeyTuple, x: np.ndarray) -> np.ndarray:
        """
        Computes the Hessian of the information at level ell
        applied to the operator x given as np.ndarray
        """
        U = self._eigen_vectors(key)
        Ud = U.conj().T
        # Hessian for a density matrix with eigenvectors
        # that are columns of U and eigenvalues applied on operator x
        hessian_on_operator = (
            U @ (self._get_hessian_matrix_elements(key) * (Ud @ x @ U)) @ Ud
        )

        return -1.0 * hessian_on_operator

    def inverse_hessian(self, key: LatticeDictKeyTuple, x: np.ndarray) -> np.ndarray:
        """
        Computes the inverse Hessian of the information at level ell
        applied to the operator x given as np.ndarray
        """
        U = self._eigen_vectors(key)
        Ud = U.conj().T
        # Inverse Hessian for a density matrix with eigenvectors
        # that are columns of U applied on operator x
        inverse_hessian = (
            U @ (1.0 / self._get_hessian_matrix_elements(key) * (Ud @ x @ U)) @ Ud
        )

        return -inverse_hessian

    def entropy_gradient(self, key: LatticeDictKeyTuple) -> np.ndarray:
        """
        Computes the gradient of the entropy
        at level ell (i.e. -gradient of information al level ell)
        """
        U = self._eigen_vectors(key)
        Ud = U.conj().T
        # gradient of entropy S for a density matrix with
        # eigenvectors that are columns of U
        return -U @ np.diag(np.log(self._eigen_values(key))) @ Ud

    def projector_to_trace_free_subspace(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the projector onto the space of traceless matrices
        obtained by tracing out range_-many sites for both sides of the operators
        (independently, i.e. either right or left) at the right and left end
        of an operator x given as np.ndarray
        """

        # trace out range sites on the left
        T_L = ptrace(x, self.range_, end="left")

        # build the projector onto the space of traceless matrices on the left
        P_L = x - np.kron(np.eye(int(2**self.range_)) * 1 / (2**self.range_), T_L)

        # use the result and trace out range_ sites on the right
        T_R = ptrace(P_L, self.range_, end="right")

        # build the projector onto the space of traceless matrices on the left and right
        P_L_R = P_L - np.kron(T_R, np.eye(int(2**self.range_)) * 1 / (2**self.range_))

        return P_L_R

    def _compute_current_operator_matrices(
        self,
        key: LatticeDictKeyTuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the current operators as matrices.
        Eq. (25) and Eq. (28) in PRX QUANTUM 5, 020352
        """
        # Note, to compute the f_n^l and g_n^l we need the rho_dict entries at level ell,
        # ell-1 and ell-2. Be sure to hand over the right rho_dict to this function
        n, ell = key
        if ell < 3:
            raise ValueError(
                "Ill-defined projectors: require at least level 3 density matrices for minimization"
            )
        else:
            # info_gradient left requires to use the info_gradient matrix at n+1/2
            n_r = key[0] + 0.5
            key_r = (n_r, ell - 1)

            n_l = key[0] - 0.5
            key_l = (n_l, ell - 1)

            g_left = self._get_commutator_of_info_gradient_with_subsystem_hamiltonian(
                key_r, "left"
            )  # 2x2 matrix is added on the left -> g_left∫
            g_right = self._get_commutator_of_info_gradient_with_subsystem_hamiltonian(
                key_l, "right"
            )

            g_left = self.projector_to_trace_free_subspace(g_left)
            g_right = self.projector_to_trace_free_subspace(g_right)

        return g_left, g_right

    def _get_relevant_info_gradient(self, key: LatticeDictKeyTuple):
        """
        This method computes - ln(\rho). In PRX QUANTUM 5, 020352
        we derive the gradient as - ln(\rho) - 1. The 1 is not necessary here since
        we compute the commutator of the gradient with the Hamiltonian and the 1 drops
        """
        return -1 * np_logm(self._density_matrix_dict[key])

    def _get_commutator_of_info_gradient_with_subsystem_hamiltonian(
        self, key: LatticeDictKeyTuple, orientation: str
    ) -> np.ndarray:
        """computes the commutator of Eq. (25) of PRX QUANTUM 5, 020352"""
        H_c = self._system_operator.subsystem_hamiltonian[key]
        if orientation == "left":
            # get the relevant parts of the information gradient
            info_gradient = self._get_relevant_info_gradient(key)
            enlarged_inf_grad = np.kron(np.eye(2**self.range_) / 2, info_gradient)

            # get the subsystem Hamiltonian
            key_ = (key[0] - 0.5 * self.range_, key[1] + self.range_)
            H = self._system_operator.subsystem_hamiltonian[key_]

            # compute the commutator
            com = commutator(enlarged_inf_grad, H.toarray())
            com_c = commutator(info_gradient, H_c.toarray())

            g = -1j * (com - np.kron(np.eye(2**self.range_) / 2, com_c))

        elif orientation == "right":
            # compute the relevant parts of the information gradient
            relevant_info_gradient = self._get_relevant_info_gradient(key)
            enlarged_inf_grad = np.kron(
                relevant_info_gradient, np.eye(2**self.range_) / 2
            )

            # get the subsystem Hamiltonian
            key_ = (key[0] + 0.5 * self.range_, key[1] + self.range_)
            H = self._system_operator.subsystem_hamiltonian[key_]

            # compute the commutators
            com = commutator(enlarged_inf_grad, H.toarray())
            com_c = commutator(relevant_info_gradient, H_c.toarray())

            g = -1j * (com - np.kron(com_c, np.eye(2**self.range_) / 2))

        else:
            raise ValueError("orientation must be 'left' or 'right' ")

        return g

    def projector_to_trace_and_current_free_subspace(
        self,
        x: np.ndarray,
        key: LatticeDictKeyTuple,
    ) -> np.ndarray:
        # project onto the trace-free space
        T_L_R = self.projector_to_trace_free_subspace(x)

        # compute g1 and g2
        g_1, g_2 = self._compute_current_operator_matrices(key)

        # project onto the orthogonal-to-g_1-space
        P1 = T_L_R - (np.trace(T_L_R @ g_1) / np.trace(g_1 @ g_1)) * g_1

        # project onto the orthogonal-to-g2-space
        P_g1_of_g2 = g_2 - (np.trace(g_2 @ g_1) / np.trace(g_1 @ g_1)) * g_1

        P12 = (
            P1
            - (np.trace(P1 @ P_g1_of_g2) / np.trace(P_g1_of_g2 @ P_g1_of_g2))
            * P_g1_of_g2
        )

        return P12

    def project_hessian_to_fixed_current(
        self, key: LatticeDictKeyTuple, x: np.ndarray
    ) -> np.ndarray:
        """
        Computes the projected hessian (PHP)
        """
        # this is HP i.e. the Hessian applied to the projected input x
        hessian_applied_to_projected_input = self.hessian(
            key, self.projector_to_trace_and_current_free_subspace(x, key)
        )

        return self.projector_to_trace_and_current_free_subspace(
            hessian_applied_to_projected_input, key
        )

    def project_gradient_to_fixed_current(self, key: LatticeDictKeyTuple) -> np.ndarray:
        """
        Computes the gradient (P Nabla_rho S)
        """
        grad = self.entropy_gradient(key)

        return self.projector_to_trace_and_current_free_subspace(grad, key)

    def precondition_to_fixed_current(
        self, key: LatticeDictKeyTuple, x: np.ndarray
    ) -> np.ndarray:
        """
        Applies preconditioning operator to each side of the equation we aim to solve.
        The best precondition would be (PHP)^+, i.e., the pseudo-inverse of (PHP).
        Our best approximation of the pseudo-inverse is (P H^-1 P)
        """
        # this is H^-1 P
        inverse_hess = self.inverse_hessian(
            key, self.projector_to_trace_and_current_free_subspace(x, key)
        )

        return self.projector_to_trace_and_current_free_subspace(inverse_hess, key)


def identity(dim: int):
    return np.eye(dim) / dim
