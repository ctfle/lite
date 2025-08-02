from __future__ import annotations

from typing import Dict
from typing import TYPE_CHECKING

import numpy as np

from local_information.core.petz_map import np_sqrt
from local_information.core.utils import get_higher_level_single_processing
from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables

if TYPE_CHECKING:
    from local_information.typedefs import SystemOperator
    from local_information.state.state import State

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()

SIGMA_X = np.array([[0, 1], [1, 0]])
SIGMA_Y = 1.0j * np.array([[0.0, -1.0], [1.0, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])


def diff_length(density_matrix: LatticeDict, operator: SystemOperator):
    """
    Computes the diffusion length of the energy distribution
    for a generic short-range Hamiltonian which corresponds to the variance of the distribution
    """

    # take the largest length scale of operators couplings
    # compute the expectation values
    # corresponds to <x^2>
    diff_length_part1 = 0.0
    # corresponds to <x>^2 --  vanishes for a uniform Hamiltonian
    diff_length_part2 = 0.0
    expt_val_H = 0.0
    expt_val_position = 0.0

    for key, rho in density_matrix.items_at_level(operator.range_):
        n = key[0]
        expt_val_n = np.trace(rho @ operator.operator[key].toarray())
        expt_val_position += n * expt_val_n
        expt_val_H += expt_val_n

    expt_val_position /= expt_val_H

    for key, rho in density_matrix.items_at_level(operator.range_):
        n = key[0]
        expt_val_n = np.trace(rho @ operator.operator[key].toarray())
        diff_length_part1 += (n - expt_val_position) ** 2 * expt_val_n
        diff_length_part2 += (n - expt_val_position) * expt_val_n

    return diff_length_part1 / expt_val_H - (diff_length_part2 / expt_val_H) ** 2


# TODO: check this function
def diff_const(density_matrix: LatticeDict, state: State, operator: SystemOperator):
    """!
    Computes the diffusion constant of the energy distribution corresponding
    to the first derivative of the variance. The observable is computed directly
    from the analytical formula of the derivative for a generic short-range Hamiltonian.
    """

    # diffusion center is set to state.anchor
    n0 = state.anchor

    # to compute the diffusion constant directly we derive the formula from the diffusion length
    # i.e. D = d_t L^2 = sum_x x**2 d_t<h_x> - d_t(sum x <h_x>) ** 2
    # -> D = sum_x x**2 d_t<h_x> - d_t(sum x <h_x>) ** 2
    # = sum_x x**2 (Tr[ h_x d_t rho_x] ) - (2*sum x <h_x>) * (sum x Tr[ h_x d_t rho_x ])
    # we call the three different parts diff_const_part1, diff_const_part2_1 and diff_const_part2_2

    # take the largest length scale of operators couplings
    # compute the expectation values
    diff_const_part1 = 0.0
    diff_const_part2_1 = 0.0
    diff_const_part2_2 = 0.0
    expt_val_H = 0.0

    # it can happen that the keys in commutator dict are not present in rho_dict
    # -> get higher level in rho_dict
    # compute the largest ell value
    if state.dyn_max_l < 2 * operator.range_:
        # update rho_dict to higher level
        for j in range(2 * operator.range_ - state.dyn_max_l):
            density_matrix += get_higher_level_single_processing(
                density_matrix, state.dyn_max_l + j
            )

    for key, rho in density_matrix.items_at_level(operator.range_):
        n = key[0]
        # compute the Hamiltonian commutators
        commutator_dict = operator.energy_current[key]
        for k in commutator_dict:
            if k in density_matrix:
                expt_val_comm = 1j * np.trace(density_matrix[k] @ commutator_dict[k])
                diff_const_part1 += (n - n0) ** 2 * expt_val_comm
                diff_const_part2_2 += (n - n0) * expt_val_comm

        expt_val_n = np.trace(rho @ operator.operator[key].toarray())
        expt_val_H += expt_val_n

        diff_const_part2_1 += 2 * (n - n0) * expt_val_n

    # the factor 0.5 is convention
    return 0.5 * (
        diff_const_part1 / expt_val_H
        - diff_const_part2_1 * diff_const_part2_2 / expt_val_H**2
    )


def onsite_operator_diff_const(
    density_matrix: LatticeDict,
    operator: LatticeDict,
    operator_current: Dict,
    n0: int,
) -> float:
    """
    Computes the diffusion constant of an onsite operator (distribution) corresponding
    to the first derivative of the variance. The observable is computed directly
    from the analytical formula of the derivative for a generic short-range Hamiltonian
    """

    ##
    # @param density_matrix density matrices
    # @param operator operators written in LatticeDict form: each key describes one element operator
    # with each of its elements
    # @param n0 Diffusion center

    # to compute the diffusion constant directly we derive the formula from the the diffusion length
    # i.e. D = d_t L^2 = sum_x x**2 ( d_t<h_x> ) - d_t(sum x <h_x>) ** 2
    # -> D = sum_x x**2 ( d_t<h_x> ) - d_t(sum x <h_x>) ** 2
    # = sum_x x**2 ( Tr[ h_x d_t rho_x] ) - (2*sum x <h_x>) * (sum x Tr[ h_x d_t rho_x ])
    # we call the three different parts diff_const_part1, diff_const_part2_1 and diff_const_part2_2

    # take the largest length scale of operators couplings
    # compute the expectation values
    diff_const_part1 = 0.0
    diff_const_part2_1 = 0.0
    diff_const_part2_2 = 0.0
    expt_val_O = 0.0

    # it can happen that the keys in commutator dict are not present in rho_dict
    # -> get higher level in rho_dict
    # compute the largest ell value

    for key, rho in density_matrix.items_at_level(0):
        n = key[0]
        # compute the Hamiltonian commutators
        commutator_dict = operator_current[key]
        for k in commutator_dict:
            if k in rho:
                expt_val_comm = 1j * np.trace(rho[k] @ commutator_dict[k])
                diff_const_part1 += (n - n0) ** 2 * expt_val_comm
                diff_const_part2_2 += (n - n0) * expt_val_comm

        expt_val_n = np.trace(rho @ operator[key].toarray())
        expt_val_O += expt_val_n

        diff_const_part2_1 += 2 * (n - n0) * expt_val_n

    diff_const = (
        diff_const_part1 / expt_val_O
        - diff_const_part2_1 * diff_const_part2_2 / expt_val_O**2
    )

    return diff_const


def energy_distribution(
    density_matrix: LatticeDict, operator: SystemOperator
) -> LatticeDict:
    """
    Computes the energy distribution for given Hamiltonian elements hamiltonian_operator_dict
    and the state given by rho_dict.
    """
    energy_dist = LatticeDict()
    for key, rho in density_matrix.items_at_level(operator.range_):
        energy_dist[key] = np.trace(rho @ operator.operator[key].toarray())

    return energy_dist


def concurrence(density_matrix: LatticeDict) -> np.ndarray:
    """
    Computes the concurrence:
    https://en.wikipedia.org/wiki/Concurrence_(quantum_computing)
    """
    gamma = 0.25 * np.kron(SIGMA_Y, SIGMA_Y)

    n_min, n_max = density_matrix.boundaries(1)
    C = np.zeros(int(n_max - n_min + 1))

    for n_count, (key, subsystem_rho) in enumerate(density_matrix.items_at_level(1)):
        R = np_sqrt(
            np_sqrt(subsystem_rho)
            @ gamma
            @ np.conjugate(subsystem_rho)
            @ gamma
            @ np_sqrt(subsystem_rho)
        )
        e, v = np.linalg.eigh(R)
        lmbda = e[::-1]
        C[n_count] = np.max([0, lmbda[0] - lmbda[1] - lmbda[2] - lmbda[3]])

    return C


def sx_correlation(density_matrix: LatticeDict) -> np.ndarray:
    """!
    Computes the one point correlation function <s_x(t)> = Tr(s_x * rho(t))
    """
    n_min, n_max = density_matrix.boundaries(0)
    sx = np.zeros(int(n_max - n_min + 1), dtype=np.complex128)

    for n_count, (key, rho) in enumerate(density_matrix.items_at_level(0)):
        sx[n_count] = 0.5 * np.trace(SIGMA_X @ rho)

    return sx


def sz_correlation(density_matrix: LatticeDict) -> np.ndarray:
    """
    Computes the one point correlation function <s_z(t)> = Trace(s_z * rho(t))
    """
    n_min, n_max = density_matrix.boundaries(0)
    sz = np.zeros(int(n_max - n_min + 1), dtype=np.complex128)

    for n_count, (key, rho) in enumerate(density_matrix.items_at_level(0)):
        sz[n_count] = 0.5 * np.trace(SIGMA_Z @ rho)

    return sz


def energy_current_mixedFieldIsing(density_matrix: LatticeDict) -> np.ndarray:
    """
    Computes the one point correlation function <s_z(t)> = Trace(s_z * rho(t))
    """
    first_term = 0.25j * np.kron(SIGMA_Z, SIGMA_Y)
    second_term = 0.25j * np.kron(SIGMA_Y, SIGMA_Z)

    n_min, n_max = density_matrix.boundaries(1)
    current = np.zeros(int(n_max - n_min), dtype=np.complex128)

    for n_count, (first_key, first_rho) in enumerate(density_matrix.items_at_level(1)):
        second_key = (first_key[0] + 1, first_key[1])
        second_rho = density_matrix[second_key]
        current[n_count] = np.trace(first_term @ first_rho) - np.trace(
            second_term @ second_rho
        )

    return current
