from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import yaml

from local_information.state.state_helper_funcs import add_higher_level_site

from local_information.lattice.lattice_dict import LatticeDict
from local_information.state.state_helper_funcs import (
    total_information,
    check_density_matrix_sequence,
    check_level_overhead,
    get_largest_dim,
)

from local_information.state.build.build_finite_state import get_finite_state
from local_information.state.build.build_repeated_elements import get_boundaries
from local_information.state.build.bulk_boundary_concatenation import get_combined
from local_information.core.utils import (
    compute_mutual_information_at_level,
    information_gradient,
)

from local_information.state.boundary.state_boundary import StateBoundary
from local_information.core.utils import (
    get_higher_level,
    ptrace,
    compute_lower_level,
)
from local_information.mpi.mpi_funcs import get_mpi_variables
from typing import TYPE_CHECKING, Any, Union, Sequence

if TYPE_CHECKING:
    from local_information.typedefs import SystemOperator

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class State:
    """
    State of the system at all times. Takes responsibility of adding sites
    when information spreads, checking convergence and updating the level.
    """

    def __init__(
        self,
        density_matrix: LatticeDict,
        case: str = "finite",
        starting_time: float = 0.0,
        loaded_state: bool = False,
        anchor: float = None,
    ):
        # boundary keys to keep track of the repeated parts at the boundaries (if necessary)
        self._state_boundary = StateBoundary.from_lattice_dict(
            density_matrix=density_matrix,
        )
        self._case = case

        # LatticeDict storing the density matrices with keys of the form (n,l)
        self.density_matrix = density_matrix

        # dynamically updated maximum level
        self.dyn_max_l = self.density_matrix.get_max_level()

        # loaded or freshly created
        self.loaded_state = loaded_state

        # compute the center of the operators
        n_values = self.density_matrix.n_at_level(self.dyn_max_l)
        if not self.loaded_state:
            self.anchor = np.mean(n_values)
        else:
            self.anchor = anchor

        self.starting_time = starting_time
        self.current_time = starting_time
        self.system_size = len(self.density_matrix) + self.dyn_max_l

    @property
    def total_information(self):
        # compute total initial information in the operators from level dyn_max_l to 0
        return total_information(self.density_matrix, self.dyn_max_l, 0)

    @property
    def total_information_at_dyn_max_l(self):
        # information at level dyn_max_l
        _, mut_info_dict = compute_mutual_information_at_level(
            self.density_matrix, self.dyn_max_l
        )
        return sum(mut_info_dict.values_at_level(self.dyn_max_l))

    @property
    def case(self) -> str:
        return self._case

    @classmethod
    def build_finite(
        cls,
        density_matrices: Sequence[np.ndarray],
        level_overhead: int = 1,
    ):
        """
        Builds the initial state given a sequence of density matrices. Assumes a finite system.

        :param Sequence[np.ndarray] density_matrices: density matrices characterising the system.
        :param int level_overhead: controls the level-overhead: the density matrices are build at a level controlled by
                the level of largest input matrix plus the level_overhead. For example if the largest matrix is
                of dimension 2 ** 4 (level 3) and level_overhead = 1 then the initial state will be constructed
                on level 3 + 1.
        """

        check_density_matrix_sequence(density_matrices)
        check_level_overhead(density_matrices, level_overhead)

        max_dim = get_largest_dim(density_matrices)
        # set the dyn_max_l to the dimension of the input + level_overhead
        dyn_max_l = max_dim - 1 + level_overhead
        state_dict, state_level = get_finite_state(
            density_matrix_sequence=density_matrices, max_l=dyn_max_l
        )

        return cls(state_dict)

    @classmethod
    def build_asymptotic(
        cls,
        bulk_density_matrices: Sequence[np.ndarray],
        boundary_density_matrices: Sequence[np.ndarray],
        level_overhead: int = 1,
    ):
        """
        Builds an asymptotically invariant state. Boundary density matrices are assumed to repeat infinitely.
        Example: The initial state deviates from the infinite temperature state on only a few sites.
        In this case, we don't need to evolve the boundaries -- they are time invariant
        (until the information spreads and they start to deviate). This principle can be extended to arbitrary
        repeating boundaries. We only need to evolve the repeating boundary density matrices once.

        :param Sequence[np.ndarray] bulk_density_matrices: density matrices characterising the bulk of the system
                (non-repeating parts).
        :param Sequence[np.ndarray] boundary_density_matrices: density matrices describing the (repeating) boundaries.
                Must be given as sequence with two elements (representing left and right boundary matrices).
        :param int level_overhead: controls the level-overhead: the density matrices are build at a level controlled by
                the level of largest input matrix plus the level_overhead. For example if the largest matrix is
                of dimension 2 ** 4 (level 3) and level_overhead = 1 then the initial state will be constructed
                on level 3 + 1.
        """

        check_density_matrix_sequence(bulk_density_matrices)
        check_density_matrix_sequence(boundary_density_matrices)

        max_dim = max(
            (
                get_largest_dim(bulk_density_matrices),
                get_largest_dim(boundary_density_matrices),
            )
        )
        # set the dyn_max_l to the dimension of the input + level_overhead
        dyn_max_l = max_dim - 1 + level_overhead
        bulk, state_level = get_finite_state(
            density_matrix_sequence=bulk_density_matrices, max_l=dyn_max_l
        )
        boundaries = get_boundaries(
            density_matrix_sequence=boundary_density_matrices, max_l=dyn_max_l
        )
        combined = get_combined(bulk=bulk, boundaries=boundaries, level=dyn_max_l)

        return cls(combined, case="asymptotic")

    @classmethod
    def build(
        cls,
        density_matrix_list: Sequence[Sequence[np.ndarray]],
        level_overhead: int = 1,
    ) -> State:
        """
        :param Sequence[Sequence[np.ndarray]] density_matrix_list: list of lists of np.ndarray indicating structure and asymptotically
                    invariant part: [[bulk], [boundary]]
        :param int level_overhead: controls the level-overhead: the density matrices are build at a level controlled by
                    the level of largest input matrix plus the level_overhead. For example if the largest matrix is
                    of dimension 2 ** 4 (level 3) and level_overhead = 1 then the initial state will be constructed
                    on level 3 + 1
        """
        bulk_density_matrices = density_matrix_list[0]
        boundary_density_matrices = density_matrix_list[1]

        check_density_matrix_sequence(bulk_density_matrices)
        check_density_matrix_sequence(boundary_density_matrices)

        if bulk_density_matrices and boundary_density_matrices:
            return State.build_asymptotic(
                bulk_density_matrices=bulk_density_matrices,
                boundary_density_matrices=boundary_density_matrices,
                level_overhead=level_overhead,
            )
        elif bulk_density_matrices and not boundary_density_matrices:
            return State.build_finite(
                density_matrices=bulk_density_matrices, level_overhead=level_overhead
            )
        else:
            raise ValueError("Empty input not allowed")

    @classmethod
    def from_product_state(cls):
        pass

    @classmethod
    def from_checkpoint(cls, folder: str) -> State:
        """load state from checkpoint"""
        p = Path(folder)
        state_filepath = p / "state.pkl"
        state_metadata = p / "state_meta_data.yaml"

        if state_filepath.is_file() and state_metadata.is_file():
            # load file
            with open(state_filepath, "rb") as file:
                rho_dict = pickle.load(file)
            # load meta_data yaml
            with open(state_metadata, "r") as file:
                loaded = yaml.safe_load(file)

            meta_data = StateMetaData(**loaded)
        else:
            raise FileNotFoundError(f"no file available in {state_filepath}")
        logger.info(f"loaded checkpoint from {folder}")

        return cls(
            rho_dict,
            case=meta_data.case,
            loaded_state=True,
            anchor=meta_data.anchor,
            starting_time=meta_data.starting_time,
        )

    def save_checkpoint(self, folder: str):
        """save state under folder"""
        # create state metadata
        meta_data = StateMetaData(
            case=self._case,
            anchor=float(self.anchor),
            starting_time=float(self.starting_time),
        )
        meta_data = asdict(meta_data)

        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)

        state_filepath = p / "state.pkl"
        state_metadata = p / "state_meta_data.yaml"
        if state_filepath.is_file() and state_metadata.is_file():
            logger.warning("files exist")

        # save as pickle
        with open(state_filepath, "wb") as file:
            pickle.dump(self.density_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)

        # save as yaml
        with open(state_metadata, "w") as file:
            yaml.dump(meta_data, file)

        logger.info(f"saved state in {state_filepath}")

    def __str__(self):
        string = ""
        for key in self.density_matrix:
            string += "\t{:<25}: {}\n".format("density matrix keys", key)
        string += "\t{:<25}: {}\n".format("case", self._case)
        string += "\t{:<25}: {}\n".format("total information", self.total_information)
        if self.loaded_state:
            string += "state loaded from previous initialization/run \n"
        return string

    def __eq__(self, other: State):
        if not isinstance(other, State):
            raise ValueError
        return self.density_matrix == other.density_matrix and np.allclose(
            self.total_information_at_dyn_max_l, other.total_information_at_dyn_max_l
        )

    def get_information_lattice(
        self,
    ) -> tuple[Union[LatticeDict, Any], Union[LatticeDict, Any]]:
        """!
        Computes the information lattice (and optionally the density matrix dictionary) on all different scales
        from the current density matrix at 'dyn_max_ell' used for the time evolution down to 0.
        Uses MPI: On all RANKs != 0 just passes
        """

        inf_dict = LatticeDict()
        work_dict = self.density_matrix.deepcopy()
        ell = self.dyn_max_l

        # get n_min n_max from rho_dict
        n_min, n_max = work_dict.boundaries(ell)
        stop = False
        while ell >= 0 and not stop:
            # TODO: workdict gets copied in this function again!
            # compute_mutual_information_at_level returns None for lower_level_dict on all RANK !=0
            # mutual_information is broad-casted
            lower_level_dict, mutual_information = compute_mutual_information_at_level(
                work_dict, ell
            )
            inf_dict += mutual_information
            if RANK == 0:
                work_dict = work_dict + lower_level_dict
                n_min -= 0.5
                n_max += 0.5
                ell -= 1
                if ell < 0:
                    stop = True
            # broadcast 'stop' to exit loop for all mpi processes
            stop = COMM.bcast(stop, root=0)

        return inf_dict, work_dict

    def get_information_current(self, operator: SystemOperator) -> dict:
        """! Computes the information current on all the sites up to dyn_max_l - range_
        (range_ is the range of the given Hamiltonian).
        Each site of the information current lattice has two elements since there are
        (at least) two currents
        """
        # compute the currents to/from higher levels up to level 'dyn_max_l - range_'
        current_dict = LatticeDict()
        if not self.dyn_max_l >= operator.range_:
            raise ValueError("Hamiltonian range larger than dyn_max_l")

        # compute the information current just where it makes sense.
        # get the density matrices on all levels
        _, rho_dict = self.get_information_lattice()
        # fast exit for all RANKs !=0
        if RANK != 0:
            return current_dict

        for ell in range(self.dyn_max_l - operator.range_ + 1):
            n_min, n_max = rho_dict.boundaries(ell)
            if (
                self._case != "finite"
                and n_min + operator.range_ > n_max - operator.range_
            ):
                # there is no current to compute here
                continue
            else:
                current_dict += information_gradient(
                    rho_dict,
                    ell,
                    n_min + operator.range_,
                    n_max - operator.range_,
                    operator.range_,
                    operator.subsystem_hamiltonian,
                )

        return current_dict

    def reduce_to_level(self, level: int, pop_boundary: bool = False):
        """reduces the level to ell"""
        if self.dyn_max_l >= level:
            for ell in range(self.dyn_max_l, level, -1):
                self.density_matrix = compute_lower_level(self.density_matrix, ell)

                if pop_boundary:
                    n_min, n_max = self.density_matrix.boundaries(ell - 1)
                    self.density_matrix.pop((n_max, ell - 1), None)
                    self.density_matrix.pop((n_min, ell - 1), None)
            self.dyn_max_l = level

    def enlarge_left(self, nr_of_sites: int):
        for _ in range(nr_of_sites):
            self._attach_site_left()

    def enlarge_right(self, nr_of_sites: int):
        for _ in range(nr_of_sites):
            self._attach_site_right()

    def _attach_site_right(self):
        """
        Enlarge the state by one site on the right end,
        proceed as follows : trace out the leftmost 'ell' spins Petz map, Petz map, update dict
                            trace out the leftmost 'ell-1' spins Petz map, Petz map, update dict
                            ...
        """
        n_max = self.density_matrix.largest_at_level(self.dyn_max_l)
        temp_dict = LatticeDict()
        for j in range(self.dyn_max_l):
            # add site at the right end
            lower_boundary_density_matrix = ptrace(
                self.density_matrix[(n_max, self.dyn_max_l)],
                self.dyn_max_l - j,
                end="left",
            )

            lower_level_n_r = n_max + 0.5 * (self.dyn_max_l - j)
            if j == 0:
                temp_dict[(lower_level_n_r + 1, j)] = (
                    self._state_boundary.lowest_level_right
                )

            temp_dict[(lower_level_n_r, j)] = lower_boundary_density_matrix
            temp_dict += add_higher_level_site(
                input_lattice=temp_dict,
                key=(lower_level_n_r, j),
                next_key=(lower_level_n_r + 1, j),
            )

        self.density_matrix[(n_max + 1, self.dyn_max_l)] = temp_dict[
            (n_max + 1, self.dyn_max_l)
        ]

    def _attach_site_left(self):
        """
        Same as '_attach_site_right' but adds site on the left end.
        """
        n_min = self.density_matrix.smallest_at_level(self.dyn_max_l)
        temp_dict = LatticeDict()
        for j in range(self.dyn_max_l):
            # add site at the left end
            lower_boundary_density_matrix = ptrace(
                self.density_matrix[(n_min, self.dyn_max_l)],
                self.dyn_max_l - j,
                "right",
            )

            lower_level_n_l = n_min - 0.5 * (self.dyn_max_l - j)

            if j == 0:
                temp_dict[(lower_level_n_l - 1, j)] = (
                    self._state_boundary.lowest_level_left
                )

            temp_dict[(lower_level_n_l, j)] = lower_boundary_density_matrix
            temp_dict += add_higher_level_site(
                input_lattice=temp_dict,
                key=(lower_level_n_l - 1, j),
                next_key=(lower_level_n_l, j),
            )

        self.density_matrix[(n_min - 1, self.dyn_max_l)] = temp_dict[
            (n_min - 1, self.dyn_max_l)
        ]

    def check_convergence(
        self, sites_to_check_left: int, sites_to_check_right: int, tolerance: float
    ):
        """
        Checks convergence to the identity towards the boundaries of the working system.
        Compare the boundary leftmost/rightmost density matrices at the lowest level
        Compare only the sites that were added in `_enlarge_system` to the
        infinite temp density matrix and delete if difference is acceptable.
        """
        for _ in range(sites_to_check_left):
            left_most = self.density_matrix.smallest_at_level(self.dyn_max_l)
            if self.norm_difference(left_most, end="left") < tolerance:
                # delete leftmost
                self.density_matrix.pop((left_most, self.dyn_max_l), None)
            else:
                break

        for _ in range(sites_to_check_right):
            right_most = self.density_matrix.largest_at_level(self.dyn_max_l)
            if self.norm_difference(right_most, end="right") < tolerance:
                # delete rightmost
                self.density_matrix.pop((right_most, self.dyn_max_l), None)
            else:
                break

        self.system_size = (
            self.density_matrix.dim_at_level(self.dyn_max_l) + self.dyn_max_l
        )

    def norm_difference(self, n_value: float, end: str):
        """
        Computes the L2 norm difference to the lowest level
        density matrix on given end in the subsystem with site index given by n_value.
        """
        if end == "left":
            lowest_level = ptrace(
                self.density_matrix[(n_value, self.dyn_max_l)], self.dyn_max_l, "right"
            )
            boundary_density_matrix = self._state_boundary.lowest_level_left
        elif end == "right":
            lowest_level = ptrace(
                self.density_matrix[(n_value, self.dyn_max_l)], self.dyn_max_l, "left"
            )
            boundary_density_matrix = self._state_boundary.lowest_level_right
        else:
            raise ValueError

        return np.linalg.norm(lowest_level - boundary_density_matrix)

    def update_dyn_max_l(self, threshold, nr_of_updates: int):
        _, temp_inf_dict = compute_mutual_information_at_level(
            self.density_matrix, self.dyn_max_l
        )
        if any([inf_value > threshold for inf_value in temp_inf_dict.values()]):
            # get higher level density matrices
            for _ in range(nr_of_updates):
                rho_dict_updated = get_higher_level(self.density_matrix, self.dyn_max_l)

                if RANK == 0:
                    self.density_matrix = rho_dict_updated
                    # update dyn_max_l
                    self.dyn_max_l += 1
                    logger.info(
                        "update dyn_max_l to {0:d}".format(
                            self.dyn_max_l,
                        )
                    )


@dataclass
class StateMetaData:
    case: str
    anchor: float
    starting_time: float
