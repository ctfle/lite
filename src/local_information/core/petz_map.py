from __future__ import annotations

import logging
import math

import numpy as np
from scipy import sparse

from functools import cached_property

from local_information.lattice.lattice_dict import LatticeKey

logger = logging.getLogger()


class PetzMap:
    """Helper class to compute the Petz map of two density matrices"""

    def __init__(
        self,
        key_A: LatticeKey,
        density_matrix_A: np.ndarray,
        key_B: LatticeKey,
        density_matrix_B: np.ndarray,
        sqrt_method: bool = True,
        precomputed_sqrt_or_log_of_density_matrix_A: np.ndarray | None = None,
        precomputed_sqrt_or_log_of_density_matrix_B: np.ndarray | None = None,
        precomputed_entropy_A: float | None = None,
        precomputed_entropy_B: float | None = None,
        smoothing_delta: float = 0.0,
    ):
        self.sqrt_method = sqrt_method
        if not math.isfinite(smoothing_delta) or smoothing_delta < 0:
            raise ValueError("smoothing_delta must be finite and >= 0")
        self.smoothing_delta = smoothing_delta

        if key_A.coord < key_B.coord:
            self.key_A = key_A
            self.key_B = key_B

            self.dens_A = density_matrix_A
            self.dens_B = density_matrix_B

        else:
            # exchange A and B
            self.key_A = key_B
            self.key_B = key_A

            self.dens_A = density_matrix_B
            self.dens_B = density_matrix_A
            # Keep all cached quantities attached to their original matrix.
            (
                precomputed_sqrt_or_log_of_density_matrix_A,
                precomputed_sqrt_or_log_of_density_matrix_B,
            ) = (
                precomputed_sqrt_or_log_of_density_matrix_B,
                precomputed_sqrt_or_log_of_density_matrix_A,
            )
            precomputed_entropy_A, precomputed_entropy_B = (
                precomputed_entropy_B,
                precomputed_entropy_A,
            )

        self._precomputed_sqrt_or_log_density_matrix_A = (
            precomputed_sqrt_or_log_of_density_matrix_A
        )
        self._precomputed_sqrt_or_log_density_matrix_B = (
            precomputed_sqrt_or_log_of_density_matrix_B
        )
        self._precomputed_entropy_A = precomputed_entropy_A
        self._precomputed_entropy_B = precomputed_entropy_B

        self.check_dimensions()
        self.check_relative_alignment()

    def get_combined_system(self):
        """
        Compute the density matrix on the combined system defined
        by the sites either in A or in B with the corresponding formula (sqrt or exp)
        """
        if self.sqrt_method:
            density_matrix_on_combined_system = self._sqrt_map()
        else:
            density_matrix_on_combined_system = self._exp_map()

        if self._overlap > 0:
            density_matrix_on_combined_system = self._correction(
                density_matrix_on_combined_system
            )

        return density_matrix_on_combined_system

    @property
    def _leftmost_A(self):
        return self.key_A.coord - self.key_A.level / 2

    @property
    def _rightmost_A(self) -> float:
        return self.key_A.coord + self.key_A.level / 2

    @property
    def _leftmost_B(self) -> float:
        return self.key_B.coord - self.key_B.level / 2

    @property
    def _rightmost_B(self) -> float:
        return self.key_B.coord + self.key_B.level / 2

    @property
    def _dimension_of_A_without_B(self) -> float:
        return self._leftmost_B - self._leftmost_A

    @property
    def _dimension_of_B_without_A(self) -> float:
        return self._rightmost_B - self._rightmost_A

    @property
    def _overlap(self):
        if self._rightmost_A >= self._leftmost_B:
            return self._rightmost_A - self._leftmost_B + 1
        else:
            return 0

    @cached_property
    def log_density_matrix_A(self):
        if (
            not self.sqrt_method
            and self._precomputed_sqrt_or_log_density_matrix_A is not None
        ):
            return self._precomputed_sqrt_or_log_density_matrix_A
        else:
            return np_logm(self.dens_A)

    @cached_property
    def log_density_matrix_B(self):
        if (
            not self.sqrt_method
            and self._precomputed_sqrt_or_log_density_matrix_B is not None
        ):
            return self._precomputed_sqrt_or_log_density_matrix_B
        else:
            return np_logm(self.dens_B)

    @cached_property
    def _eigendecomposition_A(self):
        return diagonalize(self.dens_A)

    @cached_property
    def _eigendecomposition_B(self):
        return diagonalize(self.dens_B)

    @cached_property
    def _entropy_A(self):
        if self._precomputed_entropy_A is not None:
            return self._precomputed_entropy_A
        eigenvalues, _ = self._eigendecomposition_A
        return _entropy_from_eigenvalues(eigenvalues)

    @cached_property
    def _entropy_B(self):
        if self._precomputed_entropy_B is not None:
            return self._precomputed_entropy_B
        eigenvalues, _ = self._eigendecomposition_B
        return _entropy_from_eigenvalues(eigenvalues)

    @cached_property
    def sqrt_density_matrix_A(self):
        if (
            self.sqrt_method
            and self._precomputed_sqrt_or_log_density_matrix_A is not None
        ):
            return self._precomputed_sqrt_or_log_density_matrix_A
        else:
            return _sqrt_from_eigendecomposition(*self._eigendecomposition_A)

    @cached_property
    def sqrt_density_matrix_B(self):
        if (
            self.sqrt_method
            and self._precomputed_sqrt_or_log_density_matrix_B is not None
        ):
            return self._precomputed_sqrt_or_log_density_matrix_B
        else:
            return _sqrt_from_eigendecomposition(*self._eigendecomposition_B)

    def check_relative_alignment(self):
        if (
            self._rightmost_A > self._rightmost_B
            and self._leftmost_A < self._leftmost_B
        ):
            raise AssertionError("Subsystem A contains subsystem B")
        if (
            self._rightmost_B > self._rightmost_A
            and self._leftmost_B < self._leftmost_A
        ):
            raise AssertionError("Subsystem B contains subsystem A")

    def check_dimensions(self):
        if not np.allclose(self.key_A.level + 1, np.log(len(self.dens_A)) / np.log(2)):
            raise ValueError(
                "given position tuple A does not match shape of given density matrix A"
            )
        if not np.allclose(self.key_B.level + 1, np.log(len(self.dens_B)) / np.log(2)):
            raise ValueError(
                "given position tuple B does not match shape of given density matrix B"
            )
        if self.key_A.level != self.key_B.level:
            raise ValueError(
                "matrices of different dimensions cannot be used in Petz map"
            )

    @cached_property
    def _get_density_matrix_on_overlap(self) -> np.ndarray | None:
        if self._overlap > 0:
            # density matrix of the overlap region
            # tracing out everything which is only in A
            rho_AnB_output_A = ptrace(
                self.dens_A, self._dimension_of_A_without_B, end="left"
            )
            # tracing out everything which is only in B
            rho_AnB_output_B = ptrace(
                self.dens_B, self._dimension_of_B_without_A, end="right"
            )
            # symmetric version
            rho_AnB = 0.5 * (rho_AnB_output_A + rho_AnB_output_B)
        else:
            # the two density matrices do not overlap
            rho_AnB = None

        return rho_AnB

    def _enlarge_density_matrix_on_overlap(self, density_matrix_on_overlap: np.ndarray):
        # get rho_AnB in the larger space of AB
        rho_AnB_enlarged = np.kron(
            np.kron(
                np.eye(int(2**self._dimension_of_A_without_B)),
                density_matrix_on_overlap,
            ),
            np.eye(int(2**self._dimension_of_B_without_A)),
        ) / (2 ** (self._dimension_of_A_without_B + self._dimension_of_B_without_A))

        return rho_AnB_enlarged

    @cached_property
    def _density_matrices_without_overlap(self) -> tuple[np.ndarray, np.ndarray]:
        # density matrices to compute mutual information of A and the overlap region
        rho_AuB = ptrace(self.dens_A, self._overlap, end="right")
        rho_BuA = ptrace(self.dens_B, self._overlap, end="left")
        return rho_AuB, rho_BuA

    @cached_property
    def _inverse_sqrt_and_entropy_on_overlap(self):
        """Reuse one overlap spectrum for the inverse square root and entropy."""
        eigenvalues, eigenvectors = diagonalize(self._get_density_matrix_on_overlap)
        roots = np.sqrt(eigenvalues)
        entropy = _entropy_from_eigenvalues(eigenvalues)
        if np.any(roots == 0.0):
            # Preserve failure of the previous dense inverse for singular input.
            raise np.linalg.LinAlgError("Singular matrix")
        inverse_sqrt = (eigenvectors / roots) @ eigenvectors.conj().T
        return inverse_sqrt, entropy

    def _sqrt_map(self):
        """uses the sqrt formula for the Petz map"""
        if self._overlap == 0:
            sqrt_A = np.kron(
                self.sqrt_density_matrix_A,
                np.eye(int(2**self._dimension_of_B_without_A)),
            )
            C = np.kron(np.eye(int(2**self._dimension_of_A_without_B)), self.dens_B)
            rho_AB = sqrt_A @ C @ sqrt_A
        else:
            rho_AB = self._sqrt_map_from_mutual_information()

        return rho_AB

    def _sqrt_map_from_mutual_information(self):
        """
        Compare the mutual information in (AuB, AnB) and in (AnB, BuA)
        and select or smoothly blend the two sqrt Petz map formulas.
        """
        rho_AuB, rho_BuA = self._density_matrices_without_overlap
        rho_AnB = self._get_density_matrix_on_overlap
        inverse_sqrt_overlap, entropy_overlap = self._inverse_sqrt_and_entropy_on_overlap
        information_overlap = np.log(len(rho_AnB)) - entropy_overlap
        # check the mutual information between the two different sqrt maps
        mutual_information_on_A_and_intersect = (
            (np.log(len(self.dens_A)) - self._entropy_A)
            - information(rho_AuB)
            - information_overlap
        )
        mutual_information_on_B_and_intersect = (
            (np.log(len(self.dens_B)) - self._entropy_B)
            - information(rho_BuA)
            - information_overlap
        )

        difference = (
            mutual_information_on_A_and_intersect
            - mutual_information_on_B_and_intersect
        )
        right_weight = self._right_sqrt_weight(difference)
        if right_weight == 1.0:
            return self._sqrt_candidate(inverse_sqrt_overlap, outer_right=True)
        if right_weight == 0.0:
            return self._sqrt_candidate(inverse_sqrt_overlap, outer_right=False)
        # The correction is affine, so applying it once after blending preserves
        # the same input marginals as correcting each candidate before blending.
        left = self._sqrt_candidate(inverse_sqrt_overlap, outer_right=False)
        right = self._sqrt_candidate(inverse_sqrt_overlap, outer_right=True)
        return (1.0 - right_weight) * left + right_weight * right

    def _right_sqrt_weight(self, difference: float) -> float:
        """C-infinity transition for M_left - M_right within +/- delta nats.

        Zero delta preserves the original >= branch, including its tie rule.
        Outside the band only the selected candidate needs to be constructed.
        """
        delta = self.smoothing_delta
        if delta == 0.0:
            return float(difference >= 0.0)
        if difference <= -delta:
            return 0.0
        if difference >= delta:
            return 1.0
        u = 0.5 * (1.0 + difference / delta)
        # Arithmetic can round u to an endpoint just inside the band.
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        left = math.exp(-1.0 / u)
        right = math.exp(-1.0 / (1.0 - u))
        return left / (left + right)

    def _sqrt_candidate(self, inverse_sqrt_overlap, *, outer_right: bool):
        """Contract subsystem indices without identity-extended matrices.

        a, b, c are the left-only, overlap, and right-only dimensions.
        Each intermediate has at most the size of the combined density matrix.
        """
        a = int(2**self._dimension_of_A_without_B)
        b = len(inverse_sqrt_overlap)
        c = int(2**self._dimension_of_B_without_A)
        if outer_right:
            # T_BC = sqrt(rho_BC) (rho_B^-1/2 tensor I_C).
            factor = np.einsum(
                "bcuy,ux->bcxy",
                self.sqrt_density_matrix_B.reshape(b, c, b, c),
                inverse_sqrt_overlap, optimize=True,
            )
            intermediate = np.einsum(
                "bcxy,axdz->abcdzy", factor, self.dens_A.reshape(a, b, a, b),
                optimize=True,
            )
            candidate = np.einsum(
                "abcdzy,efzy->abcdef", intermediate, factor.conj(),
                optimize=True, order="C",
            )
        else:
            # T_AB = sqrt(rho_AB) (I_A tensor rho_B^-1/2).
            factor = np.einsum(
                "abxu,uy->abxy",
                self.sqrt_density_matrix_A.reshape(a, b, a, b),
                inverse_sqrt_overlap, optimize=True,
            )
            intermediate = np.einsum(
                "abxy,yczf->abcxzf", factor, self.dens_B.reshape(b, c, b, c),
                optimize=True,
            )
            candidate = np.einsum(
                "abcxzf,dexz->abcdef", intermediate, factor.conj(),
                optimize=True, order="C",
            )
        return candidate.reshape(a*b*c, a*b*c)

    def _exp_map(self):
        """
        Petz map formula based on exponentials and logarithms.
        Discouraged since computationally approximately 8 times more expensive
        """
        if self._overlap == 0:
            log_A = np.kron(
                self.log_density_matrix_A,
                np.eye(int(2**self._dimension_of_B_without_A)),
            )
            log_B = np.kron(
                np.eye(int(2**self._dimension_of_A_without_B)),
                self.log_density_matrix_B,
            )
            rho_AB = np_expm(log_A + log_B)
        else:
            rho_AnB = self._get_density_matrix_on_overlap
            log_A = np.kron(
                self.log_density_matrix_A,
                np.eye(int(2**self._dimension_of_B_without_A)),
            )
            log_B = np.kron(
                np.eye(int(2**self._dimension_of_A_without_B)),
                self.log_density_matrix_B,
            )
            log_AnB = np.kron(
                np.kron(
                    np.eye(int(2**self._dimension_of_A_without_B)),
                    np_logm(rho_AnB),
                ),
                np.eye(int(2**self._dimension_of_B_without_A)),
            )
            rho_AB = np_expm(log_A + log_B - log_AnB)

        return rho_AB

    def _correction(self, density_matrix_on_combined_system: np.ndarray) -> np.ndarray:
        """
        An orthogonal correction map to ensure the correct subsystem density matrices
        """
        rho_AB_trace_B = ptrace(
            density_matrix_on_combined_system,
            int(self._dimension_of_B_without_A),
            end="right",
        )
        rho_AB_trace_A = ptrace(
            density_matrix_on_combined_system,
            int(self._dimension_of_A_without_B),
            end="left",
        )
        rho_AB_trace_AB = ptrace(
            rho_AB_trace_A, int(self._dimension_of_B_without_A), end="right"
        )
        density_matrix_on_overlap = self._get_density_matrix_on_overlap
        if density_matrix_on_overlap is not None:
            density_matrix_on_overlap = self._enlarge_density_matrix_on_overlap(
                density_matrix_on_overlap
            )
        # compute the corrections
        dens_correction_A = np.kron(
            (self.dens_A - rho_AB_trace_B),
            np.eye(int(2**self._dimension_of_B_without_A)),
        ) / (2**self._dimension_of_B_without_A)
        dens_correction_B = np.kron(
            np.eye(int(2**self._dimension_of_A_without_B)),
            (self.dens_B - rho_AB_trace_A),
        ) / (2**self._dimension_of_A_without_B)

        if density_matrix_on_overlap is not None:
            dens_correction_AB = density_matrix_on_overlap - np.kron(
                np.kron(
                    np.eye(int(2**self._dimension_of_A_without_B)), rho_AB_trace_AB
                ),
                np.eye(int(2**self._dimension_of_B_without_A)),
            ) / (2 ** (self._dimension_of_A_without_B + self._dimension_of_B_without_A))

            corrected_rho_AB = (
                density_matrix_on_combined_system
                + dens_correction_A
                + dens_correction_B
                - dens_correction_AB
            )
        else:
            corrected_rho_AB = (
                density_matrix_on_combined_system
                + dens_correction_A
                + dens_correction_B
            )

        return corrected_rho_AB

    def get_new_key(self) -> LatticeKey:
        # get the new key of the subsystem defined by AB
        AB_level = int(
            (self.key_B.coord + self.key_B.level / 2)
            - (self.key_A.coord - self.key_A.level / 2)
        )
        AB_n = (self.key_A.coord - self.key_A.level / 2) + AB_level / 2
        return LatticeKey(coord=AB_n, level=AB_level)


def np_logm(A: np.ndarray) -> np.ndarray:
    """Matrix logarithm"""
    e, v = diagonalize(A)
    if np.any(e.real <= 0.0):
        logger.debug(e)
        raise AssertionError("np.log produces zero or negative eigenvalue")

    log_matrix = v @ np.diag(np.log(e)) @ np.transpose(np.conjugate(v))
    return log_matrix


def np_expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential"""
    e, v = diagonalize(A)
    exp_matrix = v @ np.diag(np.exp(e)) @ np.transpose(np.conjugate(v))
    return exp_matrix


def np_sqrt(A: np.ndarray) -> np.ndarray:
    """Matrix square root"""
    e, v = diagonalize(A)
    return _sqrt_from_eigendecomposition(e, v)


def _sqrt_from_eigendecomposition(e: np.ndarray, v: np.ndarray) -> np.ndarray:
    sqrt_matrix = v @ np.diag(np.sqrt(e)) @ np.transpose(np.conjugate(v))
    return sqrt_matrix


def np_sqrt_and_entropy(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute a matrix square root and its entropy with one eigendecomposition."""
    e, v = diagonalize(A)
    return _sqrt_from_eigendecomposition(e, v), _entropy_from_eigenvalues(e)


def ptrace(
    matrix: sparse.csr_matrix | sparse.csc_matrix | np.ndarray,
    spins_to_trace_out: int,
    end: str,
) -> np.ndarray | None:
    """
    Partial trace. Can only trace out parts at the left end right end of a 1D operators.
    """
    if isinstance(matrix, (sparse.csr_matrix, sparse.csc_matrix)):
        matrix = matrix.toarray()

    assert spins_to_trace_out - int(spins_to_trace_out) == 0.0, (
        f"spins_to_trace_out must be integer but {spins_to_trace_out} was given"
    )

    spins_to_trace_out = int(spins_to_trace_out)
    if spins_to_trace_out == 0:
        return matrix
    else:
        L = int(np.log(len(matrix)) / np.log(2))

        if L - spins_to_trace_out == 0:
            return None

        bulk = 2 ** (L - spins_to_trace_out)
        boundary = 2**spins_to_trace_out

        if end == "left":
            mat_reshape = matrix.reshape(
                boundary,
                bulk,
                boundary,
                bulk,
            )
            reduced_mat = np.trace(mat_reshape, axis1=0, axis2=2)

        elif end == "right":
            mat_reshape = matrix.reshape(
                bulk,
                boundary,
                bulk,
                boundary,
            )
            reduced_mat = np.trace(mat_reshape, axis1=1, axis2=3)
        else:
            raise ValueError("input not understood: end must be 'left' or 'right'")

        return reduced_mat


def mutual_information(
    rho_AB: np.ndarray,
    rho_A: np.ndarray,
    rho_B: np.ndarray,
    rho_AnB: np.ndarray | None = None,
) -> float:
    """
    Computes mutual information of two density matrices rho_A and rho_B
    """

    if rho_AnB is None:
        mut_inf = information(rho_AB) - information(rho_A) - information(rho_B)
    else:
        mut_inf = (
            information(rho_AB)
            - information(rho_A)
            - information(rho_B)
            + information(rho_AnB)
        )

    return mut_inf


def information(rho: np.ndarray, key: tuple | None = None) -> float:
    """
    Computes the Von Neumann information of density matrix rho.
    We hand over the key to trace back possible errors
    """
    return np.log(len(rho)) - von_Neumann_entropy(rho, key)


def von_Neumann_entropy(rho: np.ndarray, key: tuple | None = None) -> float:
    """
    von Neumann entropy of rho, computing eigenvalues only.
    """

    eigenvalues = np.linalg.eigvalsh(rho)
    return _entropy_from_eigenvalues(eigenvalues, key)


def _entropy_from_eigenvalues(
    eigenvalues: np.ndarray, key: tuple | None = None
) -> float:
    """Evaluate entropy without repeating an available eigendecomposition."""
    entropy = 0.0

    for eigval in eigenvalues:
        if eigval <= 0.0:
            logger.error(
                "negative eigenvalues found while "
                f"computing the von_Neumann_entropy, eigenvalue ={eigval}"
            )
            logger.info(f"corresponding key is: {key}")
        else:
            entropy -= eigval * np.log(eigval)

    return entropy


def diagonalize(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Hermitian eigendecomposition; eigh already returns orthonormal eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return eigenvalues, eigenvectors
