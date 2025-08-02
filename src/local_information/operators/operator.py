from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict

import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from scipy import sparse

from local_information.lattice.lattice_dict import LatticeDict

if TYPE_CHECKING:
    from numbers import Number
    from local_information.typedefs import Coupling
logger = logging.getLogger()

from local_information.mpi.mpi_funcs import get_mpi_variables

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class Operator:
    def __init__(self, operator_couplings: Coupling, name=None):
        self.couplings = operator_couplings
        # compute range and L
        allowed_strings = ["x", "y", "z", "1"]
        self.range_, disorder = check_hamiltonian(operator_couplings, allowed_strings)

        if True in disorder:
            self.disorder = True
        else:
            self.disorder = False

        l = set()
        for element in operator_couplings:
            l.add(len(element[1]))
        if len(l) > 1:
            raise ValueError("lists of unequal length")
        else:
            self.L = list(l)[0]

        self.name = name

    def expectation_value(self, rho_dict: LatticeDict) -> tuple[LatticeDict, float]:
        """!
        Computes the expectation value with respect to the state of the system given by `rho_dict`
        """
        # take the largest length scale of operators couplings
        # compute the expectation values
        expt_val_dict = LatticeDict()
        expt_val = 0.0
        if RANK == 0:
            n_min, n_max = rho_dict.boundaries(self.range_)
            for n in np.arange(n_min, n_max + 1):
                key = (n, self.range_)
                value = np.trace(rho_dict[key] @ self.operator[key].toarray())
                expt_val_dict[key] = value
                expt_val += value

        expt_val_dict = COMM.bcast(expt_val_dict, root=0)
        expt_val = COMM.bcast(expt_val, root=0)

        return expt_val_dict, expt_val

    def __str__(self):
        string = f"{self.__class__.__name__} with parameters:\n"
        for key, value in self.__dict__.items():
            if key == "operator":
                continue
            else:
                string += "\t{:<25}: {}\n".format(key, value)

        # get smallest and largest key
        for ell in range(self.range_ + 1):
            n_min, n_max = self.operator.boundaries(ell)
            if n_min is None:
                continue
            else:
                string += "\t{:<25}: {}\n".format("defined between", (n_min, n_max))
                break
        list_of_ell_values = []
        for ell in range(self.range_ + 1):
            n_min, n_max = self.operator.boundaries(ell)
            if n_min is None:
                continue
            else:
                list_of_ell_values += [ell]

        string += "\t{:<25}: {}\n".format("levels", list_of_ell_values)

        return string

    @cached_property
    def operator(self) -> LatticeDict:
        """!
        Constructs all local Hamiltonians at scale of 'range' and stores
        the result as csr matrix in a lattice dict
        """
        # deepcopy operator_couplings since its modified later
        operator_couplings = deepcopy(self.couplings)
        # modify hamiltonian couplings to obtain H_n elements
        for operator_element in operator_couplings:
            factor = self.range_ - (len(operator_element[0]) - 1)

            if type(operator_element[1]) is list:
                operator_element[1] = list(np.array(operator_element[1]) / (1 + factor))
            else:
                operator_element[1] /= 1 + factor

        H_n = LatticeDict()
        for n in range(int(self.L - self.range_)):
            # construct operators for n_min= n and n_max = n_min + _ell_ +1
            ham = construct_hamiltonian(
                n + self.range_ + 1, operator_couplings, n_min=n
            )
            H_n[(n + self.range_ / 2, self.range_)] = ham.tocsr()

        return H_n

    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False

        return self.couplings == other.couplings

    def __add__(self, other: Operator) -> Operator:
        """
        Computes the sum of Operators. Creates a new operator using the couplings
        of `self` and `other`.
        """
        couplings = self.couplings + other.couplings
        return Operator(couplings)

    def __sub__(self, other: Operator) -> Operator:
        """
        Same as add but with inverted (-1) couplings in `other`.
        """
        subtract_couplings = []
        for _, coupling_element in enumerate(other.couplings):
            inverse_coupling = [-1 * element for element in coupling_element[1]]

            subtract_couplings += [[coupling_element[0], inverse_coupling]]

        couplings = self.couplings + subtract_couplings
        return Operator(couplings)

    def __mul__(self, value: Number):
        """
        Scalar multiplication
        """
        for key, op in self.operator.items():
            updated_op = op * value
            self.operator[key] = updated_op
        # update operator couplings
        couplings = []
        for _, coupling_element in enumerate(self.couplings):
            inverse_coupling = [element * value for element in coupling_element[1]]

            couplings += [[coupling_element[0], inverse_coupling]]
        self.couplings = couplings

        pass

    def __rmul__(self, value: Number):
        return self.__mul__(value)


@dataclass
class HamiltonianData:
    name: str
    max_l: int
    L: int


def legit_string(string: str, string_list: list[str]) -> bool:
    return string in string_list


def to_lst(tpl: tuple) -> list:
    return list(to_lst(i) if isinstance(i, tuple) else i for i in tpl)


def setup_element(value: Number, j: int, rng: int) -> list:
    el = [value]
    for h in range(rng):
        el += [j + h]
    return el


def check_hamiltonian(
    operator_couplings: Coupling, allowed_strings: list
) -> tuple[int, Coupling]:
    """!
    Check if input Hamiltonian is of right form.
    """
    range_ = 0

    disorder = []
    for h_element in operator_couplings:
        for string in h_element[0]:
            if not any(
                list(
                    map(legit_string, [string] * len(allowed_strings), allowed_strings)
                )
            ):
                raise ValueError(
                    "input not understood: "
                    + h_element[0]
                    + " is not a valid input for an Operator coupling"
                )

        if isinstance(h_element[1], list):
            disorder.append([not all(x == h_element[1][0] for x in h_element[1])])

        # get the max operators range
        if len(h_element[0]) - 1 > range_:
            range_ = len(h_element[0]) - 1

    return range_, disorder


def construct_operator_dict(
    operator_couplings: Coupling, max_l: int, range_: int, system_size: int
) -> LatticeDict:
    """!
    Constructs all Hamiltonians up to max_l + range and stores the result as csr matrix in a lattice dict.
    """
    ##
    # @param rho_dict the input LatticeDict of density matrices
    # @param operator_couplings all operators couplings
    # @param max_l maximum correlation length to be used for time evolution
    # @param dyn_max_l used to compute the operators size
    # @param anchor parameter to anchor the build of the Hamiltonian. Default is 0.
    # This means the 0th index in the Hamiltonian coupling list will be used for the density matrix with key (n=0,l=0).
    # For asymptotic invariant systems anchor is typically not zero.
    #

    hamiltonian_dict = LatticeDict()

    for ell in range(max_l + range_ + 1):
        # construct operators at _ell_ +1: _ell_=0 means single spin operators,
        # therefore we need a Hamiltonian for that spin

        for n in range(int(system_size - ell)):
            # construct operators for n_min= n and n_max = n_min + _ell_ +1
            ham = construct_hamiltonian(n + ell + 1, operator_couplings, n_min=n)
            hamiltonian_dict[(n + ell / 2, ell)] = ham.tocsr()

    return hamiltonian_dict


def construct_hamiltonian(
    n_max: int, operator_couplings: Coupling, n_min: int = 0
) -> hamiltonian:
    """!
    Constructs the Hamiltonian on the operators defined from the sites n_min to n_max.
    """
    ##
    # @param n_max maximum site to construct the Hamiltonian
    # @param operator_couplings all Hamiltonian couplings in list form.
    # Each element of the list is a list itself containing a
    # string that characterizes the coupling type ('x', 'y', 'z', '1')
    # For instance: [['z',h],['zz',J]]. J and h can be lists themselves
    # @param n_min minimum site to construct the Hamiltonian. Default is 0
    #

    # quspin basis object
    basis = spin_basis_1d(L=int(n_max - n_min), pauli=True)

    # get all the couplings in 'quspin-readable' form
    couplings = get_couplings(to_lst(operator_couplings), n_max, n_min)

    return hamiltonian(
        couplings,
        dynamic_list=[],
        basis=basis,
        dtype=np.complex128,
        check_symm=False,
        check_herm=False,
    )


def get_couplings(operator_couplings: Coupling, n_max: int, n_min: int) -> Coupling:
    """!
    Computes all operators couplings in quspin-readable form from the input list operator_couplings
    """
    couplings = []
    for h_element in operator_couplings:
        # check length
        range_of_element = len(h_element[0])

        if type(h_element[1]) is list or type(h_element[1]) is np.ndarray:
            couplings += [
                [
                    h_element[0],
                    [
                        setup_element(h_element[1][int(n_min + j)], j, range_of_element)
                        for j in range(int((n_max - n_min) - (range_of_element - 1)))
                    ],
                ]
            ]
        else:
            couplings += [
                [
                    h_element[0],
                    [
                        setup_element(h_element[1], j, range_of_element)
                        for j in range(int((n_max - n_min) - (range_of_element - 1)))
                    ],
                ]
            ]
    return couplings


def compute_HH_commutator(H_n_dict: LatticeDict, key: tuple) -> LatticeDict:
    """!
    Function to compute the commutator of the local Hamiltonian with key=(n,ell)
    """
    ##
    # @param H_n_dict Hamiltonian in dict form: each key corresponds to one element H^ell_n
    # @param key for which to compute the commutator, i.e., key -> h^ell_n to compute [H,h^ell_n]
    #

    # range of Hamiltonian + range of operator + 1
    n, ell = key
    reference_hamiltonian = H_n_dict[key]
    commutator_dict = LatticeDict()
    for l_ in range(2 * int(ell) + 1):
        k = (n - ell + l_, ell)

        if k in H_n_dict:
            if ell - l_ > 0:
                # operators left of reference Hamiltonian - add sites on the right
                # in operators and on the left in reference Hamiltonian to be able to compute the commutator
                a = ell - l_
                hamiltonian = add_spins(H_n_dict[k], a, "right")
                ref_ham = add_spins(reference_hamiltonian, a, "left")
                commutator_dict[(n - a + a / 2, ell + a)] = (
                    hamiltonian @ ref_ham - ref_ham @ hamiltonian
                )

            elif ell - l_ == 0:
                # operators are exactly the same as reference Hamiltonian -- continue
                continue

            elif ell - l_ < 0:
                # operators right of reference Hamiltonian
                # - add sites on the left in operators and on the right in reference Hamiltonian
                a = -(ell - l_)
                hamiltonian = add_spins(H_n_dict[k], a, "left")
                ref_ham = add_spins(reference_hamiltonian, a, "right")
                commutator_dict[(n + a - a / 2, ell + a)] = (
                    hamiltonian @ ref_ham - ref_ham @ hamiltonian
                )
        else:
            continue

    return commutator_dict


def compute_H_onsite_operator_commutator(H_n_dict, operator_dict, key, range_):
    """!
    Function to compute the commutator of the Hamiltonian with an onsite
    Operator at key=(n,ell)
    """
    ##
    # @param H_n_dict Hamiltonian in dict form: each key corresponds to one element H^ell_n
    # @param operator_dict Same as H_n_dict for the onsite operator of interest
    # @param key for which to compute the commutator, i.e., key -> h^ell_n to compute [H,h^ell_n]
    # @param range_ range of the Hamiltonian

    # key of the operator: where to compute the commutator with the Hamiltonian
    n, ell = key
    reference_operator = operator_dict[key]
    commutator_dict = LatticeDict()

    # in general: range of Hamiltonian + range of operator + 1
    for l_ in range(int(range_) + 1):
        k = (n - range_ / 2 + l_, range_)

        if k in H_n_dict:
            # add sites on the right and the left to ensure same dimensions
            a_l = range_ - l_
            a_r = l_

            ref_operator = add_spins(reference_operator, a_l, "left")
            ref_operator = add_spins(ref_operator, a_r, "right")
            commutator_dict[k] = H_n_dict[k] @ ref_operator - ref_operator @ H_n_dict[k]
        else:
            continue

    return commutator_dict


def add_spins(operator, number, orientation: str):
    """!
    Adds identity at given orientation
    """
    ##
    # @param hamiltonian Hamiltonian in sparse format
    # @param number number of 2x2 identities to enlarge the Hilbert-space
    # @param orientation allowed values 'left' or 'right' to add the identities
    #
    if isinstance(operator, np.ndarray):
        pass
    else:
        operator = operator.toarray()

    if orientation == "left":
        for j in range(int(number)):
            operator = np.kron(np.eye(2, dtype=np.complex128), operator)

    elif orientation == "right":
        for j in range(int(number)):
            operator = np.kron(operator, np.eye(2, dtype=np.complex128))

    return operator


def check_lindbladian(
    jump_couplings: Coupling, allowed_strings: list
) -> tuple[int, list, Coupling]:
    """!
    Check if input for Lindbladian is of right form
    """
    range_ = 0
    type_list = []

    disorder = []
    for h_element in jump_couplings:
        for string in h_element[0]:
            if not any(
                list(
                    map(legit_string, [string] * len(allowed_strings), allowed_strings)
                )
            ):
                raise AssertionError(
                    "input not understood: "
                    + h_element[0]
                    + " is not a valid input for a Hamiltonian coupling"
                )

        if isinstance(h_element[1], list):
            disorder.append([not all(x == h_element[1][0] for x in h_element[1])])

        # get the max operators range
        if len(h_element[0]) - 1 > range_:
            range_ = len(h_element[0]) - 1

        if len(h_element[0]) == 1:
            type_list += [h_element[0]]

    type_list = list(set(type_list))

    return range_, type_list, disorder


def construct_lindbladian_dict(
    jump_couplings: Coupling, max_l: int, range_: int, system_size: int
) -> Dict:
    """!
    Constructs a dictionary that holds the information which on-site Lindblad operator to apply where
    """
    lindbladian_dict = dict()
    for ell in range(max_l + range_ + 1):
        # contains ell + 1  spins: ell = 0 means single spin operators,
        for n in range(int(system_size - ell)):
            # transform the input jump_couplings into the form n_min= n and n_max = n_min + _ell_ +1
            # i.e. all the physical sites that are included in the triangle with (n + ell / 2, ell) as top
            lindbald_identifier = construct_lindbladian_id(
                n + ell + 1, jump_couplings, n_min=n
            )
            lindbladian_dict[(n + ell / 2, ell)] = lindbald_identifier

    return lindbladian_dict


def construct_lindbladian_id(
    n_max: int, jump_couplings: Coupling, n_min: int = 0
) -> list:
    """!
    Construct the Lindbladian id for the operators from n_min to n_max.
    """

    # initialize as list with n_max - n_min entries
    id_list = [None for _ in range(n_max - n_min)]
    for jump_term in jump_couplings:
        jump_type = jump_term[0]
        for n in range(n_min, n_max):
            jump_value = jump_term[1][n]
            if jump_value == 0 or jump_value is None:
                continue
            else:
                if id_list[n - n_min] is None:
                    id_list[n - n_min] = [(jump_type, jump_value)]
                else:
                    id_list[n - n_min] += [(jump_type, jump_value)]

    return id_list


def setup_onsite_L_operators(max_l: int, range_: int, type_list: list) -> LatticeDict:
    """!
    Construct all basic onsite Lindblad operators up to level max_l + range_. Keys are tuples of length 3:
    (extend, m, type) where m is the site where the operator acts, extend: over which we construct the operator,
    type: what kind of operator. I.e. generates a LatticeDict with all possible single particle Lindblad operators
    projected to larger Hilbert spaces
    """
    L_operators = LatticeDict()

    basic_operators = {
        "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
        "+": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
        "-": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
        "1": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    }

    for ell in range(max_l + range_ + 1):
        for m in range(ell + 1):
            for tpe in type_list:
                if ell == 0:
                    L_operators[(ell, m, tpe)] = sparse.csr_matrix(basic_operators[tpe])
                else:
                    if m == 0:
                        operator = np.kron(
                            basic_operators[tpe],
                            np.eye(2 ** (ell - m), dtype=np.complex128),
                        )
                    elif m == ell:
                        operator = np.kron(
                            np.eye(2**m, dtype=np.complex128), basic_operators[tpe]
                        )
                    else:
                        operator = np.kron(
                            np.eye(2**m, dtype=np.complex128),
                            np.kron(
                                basic_operators[tpe],
                                np.eye(2 ** (ell - m), dtype=np.complex128),
                            ),
                        )

                    L_operators[(ell, m, tpe)] = sparse.csr_matrix(operator)

    return L_operators
