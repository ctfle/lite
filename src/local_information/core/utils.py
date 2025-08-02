from __future__ import annotations

import logging

import numpy as np
from itertools import permutations
from enum import Enum

from local_information.core.petz_map import (
    PetzMap,
    np_logm,
    np_sqrt,
    ptrace,
    information,
)
from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi import MultiProcessing
from local_information.mpi.mpi_funcs import get_mpi_variables

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


@MultiProcessing(shift=1)
def get_higher_level(
    input_dict: LatticeDict,
    level: int,
    sqrt_method: bool = True,
) -> LatticeDict:
    """
    Computes all density matrices at level ell+1 given density matrices
    at level ell in between n_min to n_max
    """
    return get_higher_level_single_processing(
        input_dict, level, sqrt_method=sqrt_method
    )


def get_higher_level_single_processing(
    input_dict: LatticeDict,
    level: int,
    sqrt_method: bool = True,
) -> LatticeDict:
    """
    Computes the density matrices one level higher than `ell` using the Petz map
    in between the boundaries [n_min, n_max] (n_max included).
    """
    changes = LatticeDict()
    # Only proceed if there are at least two sites
    # Single site at given level can happen during MPI
    if not input_dict.has_single_entry_at_level(level):
        if sqrt_method:
            # precompute all sqrts
            precomp_dict = LatticeDict()
            for key, density_matrix in input_dict.items_at_level(level):
                precomp_dict[key] = np_sqrt(density_matrix)

        else:
            # precompute the logm of all relevant density matrices
            precomp_dict = LatticeDict()
            for key, density_matrix in input_dict.items_at_level(level):
                precomp_dict[key] = np_logm(density_matrix)

        keys_at_ell = list(input_dict.keys_at_level(level))
        # do the petz map
        for key, next_key in zip(keys_at_ell[:-1], keys_at_ell[1:]):
            precomp_dens_A = precomp_dict[key]
            precomp_dens_B = precomp_dict[next_key]

            dens_A = input_dict[key]
            dens_B = input_dict[next_key]

            petz_map = PetzMap(
                key_A=key,
                key_B=next_key,
                density_matrix_A=dens_A,
                density_matrix_B=dens_B,
                precomputed_sqrt_or_log_of_density_matrix_A=precomp_dens_A,
                precomputed_sqrt_or_log_of_density_matrix_B=precomp_dens_B,
                sqrt_method=sqrt_method,
            )
            density_matrix_on_combined_system = petz_map.get_combined_system()
            new_key = petz_map.get_new_key()
            changes[(new_key.n, new_key.level)] = density_matrix_on_combined_system
    return changes


def compute_mutual_information_at_level(
    rho_dict: LatticeDict,
    level: int,
) -> tuple[LatticeDict, LatticeDict]:
    """
    Compute the mutual information for each site of the information lattice.
    Returns the mutual information together with the lower density matrices at level 'ell - 1'.
    """
    work_dict = rho_dict.deepcopy()
    # compute 2 lower levels and stores them in a separate LatticeDict
    inf_dict = compute_von_Neumann_information(work_dict, level)

    # scatter `ell` from root
    work_level = None
    if RANK == 0:
        work_level = level
    work_level = COMM.bcast(work_level, root=0)

    for i in range(2):
        if work_level - i - 1 >= 0:
            lower_level_dict = compute_lower_level(work_dict, level - i)
            # compute von Neumann information at level ell -i -1
            work_dict += lower_level_dict
            inf_dict_lower = compute_von_Neumann_information(work_dict, level - i - 1)
            if RANK == 0:
                inf_dict += inf_dict_lower

    if RANK == 0:
        work_dict.kill_all_except(level - 1)
        mut_inf_dict = assemble_mutual_information_from_dict(inf_dict, level)
    else:
        mut_inf_dict = None
        work_dict = None
    mut_inf_dict = COMM.bcast(mut_inf_dict, root=0)

    return work_dict, mut_inf_dict


def compute_lower_level(rho_dict: LatticeDict, ell: int) -> LatticeDict:
    """
    Computes the density matrices at level ell - 1 if no lower level exists,
    returns an empty dict. Computes just one of the lower leve density matrices
    for ech LatticeDict node. This works if all the keys between the boundaries
    have density matrices associated. If this is not the case as for example when
    initializing a `State` use `compute_lower_level_sparse`.
    """
    lower_level_dict = LatticeDict()
    if ell != 0:
        for ind, (key, r_AB) in enumerate(rho_dict.items_at_level(ell)):
            if r_AB is not None:
                # to avoid double computations, recycle traced matrices
                if ind == 0:
                    # trace out leftmost site
                    r_B = ptrace(r_AB, 1, end="left")
                    # trace out rightmost site
                    r_A = ptrace(r_AB, 1, end="right")
                    # trace out right- and leftmost site
                    # update rho_dict on level ell-1
                    lower_level_dict[(key[0] - 0.5, ell - 1)] = r_A
                    lower_level_dict[(key[0] + 0.5, ell - 1)] = r_B
                else:
                    # trace out leftmost site
                    r_B = ptrace(r_AB, 1, end="left")
                    # update rho_dict on level ell-1
                    lower_level_dict[(key[0] + 0.5, ell - 1)] = r_B

    return lower_level_dict


def compute_lower_level_sparse(rho_dict: LatticeDict, ell: int) -> LatticeDict:
    """
    Same as `compute_lower_level` but computes *both* lower level density matrices
    associated with each LatticeDict node.
    """
    lower_level_dict = LatticeDict()
    if ell != 0:
        for ind, (key, r_AB) in enumerate(rho_dict.items_at_level(ell)):
            if r_AB is not None:
                # trace out leftmost site
                r_B = ptrace(r_AB, 1, end="left")
                # trace out rightmost site
                r_A = ptrace(r_AB, 1, end="right")
                # trace out right- and leftmost site
                # update rho_dict on level ell-1
                lower_level_dict[(key[0] - 0.5, ell - 1)] = r_A
                lower_level_dict[(key[0] + 0.5, ell - 1)] = r_B

    return lower_level_dict


def assemble_mutual_information_from_dict(
    inf_dict: LatticeDict, ell: int
) -> LatticeDict:
    """
    Compute mutual information from LatticeDict of von Neumann information at level ell
    """
    mut_info = LatticeDict()
    if ell >= 2:
        for key, info in inf_dict.items_at_level(ell):
            # compose mutual information
            mut_info[key] = (
                info
                - inf_dict[(key[0] - 0.5, ell - 1)]
                - inf_dict[(key[0] + 0.5, ell - 1)]
                + inf_dict[(key[0], ell - 2)]
            )
    elif ell == 1:
        for key, info in inf_dict.items_at_level(ell):
            mut_info[key] = (
                info
                - inf_dict[(key[0] - 0.5, ell - 1)]
                - inf_dict[(key[0] + 0.5, ell - 1)]
            )
    else:
        for key, info in inf_dict.items_at_level(ell):
            mut_info[key] = info

    return mut_info


@MultiProcessing(shift=0)
def compute_von_Neumann_information(input_dict: LatticeDict, level: int) -> LatticeDict:
    return compute_von_Neumann_information_single_processing(
        rho_dict=input_dict, level=level
    )


def compute_von_Neumann_information_single_processing(
    rho_dict: LatticeDict, level: int
) -> LatticeDict:
    """
    Computes von Neumann information at given level
    """
    inf_dict = LatticeDict()
    if level >= 0:
        for key, density_matrix in rho_dict.items_at_level(level):
            inf_dict[key] = information(density_matrix, key=key)

    return inf_dict


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def anti_commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A


def compute_commutator(
    rho_dict: LatticeDict,
    key: tuple,
    range_: int,
    subsystem_hamiltonian: LatticeDict,
    orientation: str,
) -> np.ndarray:
    """!
    Computes the commutator of the density matrix in the enlarged system (left/right) with the corresponding
    subsystem Hamiltonian. We subtract the commutator in the subsystem defined by 'key' to extract the terms
    associated with coupling to the left/right.
    """
    if key not in rho_dict:
        # ensures the while loop terminates
        raise ValueError("given key not in LatticeDict")

    if orientation not in ["left", "right"]:
        raise ValueError("orientation must be 'left' or 'right'")

    reduce_range = 0
    key_ = None
    search_for_key = True
    distance = range_
    DM = None
    while search_for_key:
        distance = range_ - reduce_range

        if orientation == "left":
            key_ = (key[0] - 0.5 * distance, key[1] + distance)
        else:
            key_ = (key[0] + 0.5 * distance, key[1] + distance)

        try:
            DM = rho_dict[key_]
            search_for_key = False
        except KeyError:
            search_for_key = True
            reduce_range += 1

    DM_c = rho_dict[key]
    H = subsystem_hamiltonian[key_]
    H_c = subsystem_hamiltonian[key]

    com = ptrace(commutator(H.toarray(), DM), distance, end=orientation)
    com_c = commutator(H_c.toarray(), DM_c)

    return -1j * (com - com_c)


def information_gradient(
    rho_dict: LatticeDict,
    ell: int,
    n_min: float,
    n_max: float,
    range_: int,
    subsystem_hamiltonian: LatticeDict,
) -> LatticeDict[tuple[float, int], np.ndarray]:
    """!
    Computes the information gradient values on level ell. Separates the three possible cases:
    ell=0, ell=1 and ell>1 (which all have different formulas).
    """
    inf_current = LatticeDict()
    for n in list(np.arange(n_min, n_max + 1)):
        r_AB = rho_dict[(n, ell)]
        info_gradient = np_logm(r_AB)

        if ell >= 1:
            r_B = rho_dict[(n + 0.5, ell - 1)]
            r_A = rho_dict[(n - 0.5, ell - 1)]
            info_gradient -= np.kron(np.eye(2), np_logm(r_B)) + np.kron(
                np_logm(r_A), np.eye(2)
            )

        if ell >= 2:
            r_AnB = rho_dict[(n, ell - 2)]
            info_gradient -= np.kron(np.kron(np.eye(2), np.eye(len(r_AnB))), np.eye(2))

        commutator_left = compute_commutator(
            rho_dict, (n, ell), range_, subsystem_hamiltonian, orientation="left"
        )
        commutator_right = compute_commutator(
            rho_dict, (n, ell), range_, subsystem_hamiltonian, orientation="right"
        )

        current_left = np.trace(info_gradient @ commutator_left)
        current_right = np.trace(info_gradient @ commutator_right)
        inf_current[(n, ell)] = np.array([current_left, current_right])

    return inf_current


def one_shift(rho_dict: LatticeDict, alpha: float = 1.0) -> LatticeDict:
    """
    creates a LatticeDict with the same keys as the input
    and diagonal matrices of trace alpha (default 1)
    """
    keys = list(rho_dict.keys())
    dims = list(map(len, list(rho_dict.values())))
    shifts = [alpha * np.diag(np.ones(dims[j]) / dims[j]) for j in range(len(dims))]
    return LatticeDict.from_list(keys, shifts)


def push_keys(rho_dict: LatticeDict, number: float) -> LatticeDict:
    return_dict = LatticeDict()
    for key in list(rho_dict):
        n = key[0]
        new_n = n + number
        new_key = (new_n, key[1])
        return_dict[new_key] = rho_dict[key]

    return return_dict


def arctanh(x: float) -> float:
    return 0.5 * np.log(x + 1.0) - 0.5 * np.log(1 - x)


def update_and_scatter(
    rho_dict: LatticeDict, n_min: float, n_max: float, ell: int
) -> LatticeDict:
    higher_level = get_higher_level(rho_dict, n_min, n_max, ell)
    if RANK == 0:
        rho_dict += higher_level
    rho_dict = COMM.bcast(rho_dict, root=0)
    return rho_dict


def align_to_level(density_matrix: LatticeDict, level: int) -> LatticeDict:
    input_level = density_matrix.get_max_level()
    if input_level > level:
        # reduce to level
        ell = input_level
        while ell > level:
            density_matrix = compute_lower_level(density_matrix, ell)
            ell -= 1

    elif input_level < level:
        # increase level
        ell = input_level
        while ell < level:
            n_min, n_max = density_matrix.boundaries(ell)
            density_matrix = get_higher_level(
                density_matrix, n_min=n_min, n_max=n_max, ell=ell
            )
            ell += 1

    return density_matrix


def bin_conv(integer, L):
    return bin(integer)[2:].zfill(L)


def int_conv(string):
    return int(string, 2)


def sequence_constructor(lst: list | np.ndarray, subsequence: int):
    lst = list(lst)
    sequence_list = []
    n = 0

    while n + subsequence < len(lst):
        sequence_list.append(lst[n : n + subsequence])
        n += 1

    return sequence_list


def splitter(string: str, split_elements: list):
    """
    Function that splits a string in two parts.
    Used for splitting basis states of the form '10010110111 ...' for partial tracing
    """
    part_one = ""
    for j in split_elements:
        part_one += string[int(j)]

    if len(string) - 1 == int(split_elements[-1]):
        part_two = string[: int(split_elements[0])]
    else:
        part_two = (
            string[: int(split_elements[0])] + string[int(split_elements[-1]) + 1 :]
        )

    return [part_one, part_two]


def partial_trace(
    dens_mat: np.ndarray, basis: list, sub_sys_A: list
) -> np.ndarray | int:
    """
    Computes partial trace on the sites not contained in sub_sys_A.
    This function is able to trace out any part of the operators.
    Note that the function ptrace can only trace out parts at the left/right end of a region.
    To be able to do so partial_trace requires a basis as argument
    """

    L = int(np.log(len(basis)) / np.log(2))
    if type(basis[0]) is not str:
        map_L = [int(L) for _ in range(len(basis))]
        basis = list(map(bin_conv, basis, map_L))
    else:
        pass

    # split all elements in basis in two parts
    map_sys = [sub_sys_A for _ in range(len(basis))]
    subsys_el = np.array(list(map(splitter, basis, map_sys)))

    if subsys_el.T[1, 0] == "":
        return dens_mat

    elif subsys_el.T[0, 0] == "":
        return 0

    else:
        trace_out = np.array(list(map(int_conv, subsys_el.T[1])))
        reduced_basis = np.array(list(map(int_conv, subsys_el.T[0])))

        trace_out_copy = np.copy(trace_out)
        index_pairs = []

        while len(trace_out_copy) > 0:
            c = list(np.nonzero(trace_out == trace_out_copy[0])[0])
            cc = list(np.nonzero(trace_out_copy == trace_out_copy[0])[0])
            per = permutations(c, 2)
            index_pairs += [p for p in per]
            index_pairs += list(map(tuple, list(np.array([c, c]).T)))
            trace_out_copy = np.delete(trace_out_copy, cc)

        new_basis_states = list(set(reduced_basis))[::-1]

        basis_ind = list(np.arange(len(new_basis_states)))
        basis_dict = dict(zip(new_basis_states, basis_ind))
        rdm = np.zeros(
            (len(new_basis_states), len(new_basis_states)), dtype=np.complex128
        )

        for r_c_element in index_pairs:
            # get the value in the old basis
            val = dens_mat[r_c_element[0], r_c_element[1]]

            # r_element and c_element are ints describing a certain state
            r_element = reduced_basis[r_c_element[0]]
            c_element = reduced_basis[r_c_element[1]]

            # get where these states are located in the new reduced basis
            ind_r_element = basis_dict[r_element]
            ind_c_element = basis_dict[c_element]

            # add the corresponding value
            rdm[ind_r_element, ind_c_element] += val

        return rdm


class Status(Enum):
    Stop = 0
    Continue = 1


class Trigger:
    """Helper class for control flow of the time evolution procedure."""

    def __init__(self):
        self._status = Status.Continue

    def pull(self):
        self._status = Status.Stop

    def reset(self):
        self._status = Status.Continue

    @property
    def status(self) -> Status:
        return self._status
