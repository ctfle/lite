from __future__ import annotations

import logging
import pickle
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables
from local_information.operators.operator import (
    construct_operator_dict,
    construct_lindbladian_dict,
    setup_onsite_L_operators,
    compute_HH_commutator,
    compute_H_onsite_operator_commutator,
    check_lindbladian,
    Operator,
    HamiltonianData,
)

if TYPE_CHECKING:
    from local_information.typedefs import Coupling

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class Lindbladian(Operator):
    """!
    Class to  construct Lindbladians (for on-site dissipators) organized as LatticeDict to compute and time evolve
    subsystem density matrices based on the Petz map and the corresponding information lattice and currents. Stores
    the couplings, the `operator` (decomposed into locel terms and stored as LatticeDict) and all the
    `subsystem_hamiltonian`s as well as `subsystem_lindbladians`. Note: there is a difference between `operator` and
    `subsystem_hamiltonian`. Summation of all values of the `operator` LatticeDict yields the Hamitlonian of the entire
    system. This is not the case for `subsystem_hamiltonian` (since subsystems overlap different terms would be counted
    multiple times).
    """

    def __init__(
        self, max_l: int, hamiltonian_couplings: Coupling, jump_couplings: Coupling
    ):
        ## Maximum correlation length. As long as dyn_max_l < max_l, the exact Petz map will be used
        self.max_l = max_l

        super().__init__(hamiltonian_couplings, name="Lindbladian")

        ## Hamiltonian couplings
        self.hamiltonian_couplings = hamiltonian_couplings

        ## Couplings describing the Lindblad jump coupling.
        self.jump_couplings = jump_couplings

        allowed_strings = ["x", "y", "z", "1", "+", "-"]
        lindblad_range, type_list, lindblad_disorder = check_lindbladian(
            jump_couplings, allowed_strings
        )
        if lindblad_range != 0:
            raise ValueError(
                "Lindblad operators must be on-site for this implementation"
            )

        if self.disorder and True in lindblad_disorder:
            self.disorder = True
        else:
            self.disorder = False

        ## LatticeDict object storing all relevant subsystem Hamiltonians
        self.subsystem_hamiltonian = construct_operator_dict(
            hamiltonian_couplings, self.max_l, self.range_, self.L
        )

        ## LatticeDict object storing all information where to
        # apply onsite Lindblad operators for any lattice point \f$ (n, \ell) \f$
        self.lindbladian_dict = construct_lindbladian_dict(
            jump_couplings, self.max_l, self.range_, self.L
        )

        ## LatticeDict containing all basic onsite lindblad operators
        self.L_operators = setup_onsite_L_operators(self.max_l, self.range_, type_list)

    def operator_current(
        self, operator: Operator
    ) -> dict[tuple[float, int], LatticeDict]:
        """Computes the commutator of the Hamiltonian with each element of an onsite (!) operator."""
        op_current: dict[tuple[float, int], LatticeDict] = {}
        for key in operator.operator.keys_at_level(operator.range_):
            op_current[key] = compute_H_onsite_operator_commutator(
                H_n_dict=self.operator,
                operator_dict=operator.operator,
                key=key,
                range_=self.range_,
            )
        return op_current

    @cached_property
    def energy_current(self) -> dict[tuple[float, int], LatticeDict]:
        """! Computes the commutator of the Hamiltonian with each of its decomposed local elements."""
        e_current = dict()
        for key in self.operator.keys_at_level(self.range_):
            e_current[key] = compute_HH_commutator(self.operator, key)

        return e_current

    @classmethod
    def from_checkpoint(cls, folder: str):
        """
        load Lindbladian from checkpoint.
        """
        p = Path(folder)
        filepath = p / "lindbladian.pkl"
        metadata = p / "lindbladian_config.yaml"

        if filepath.is_file() and metadata.is_file():
            # load file
            with open(filepath, "rb") as file:
                [hamiltonian_couplings, jump_couplings] = pickle.load(file)
            # load meta_data yaml
            with open(metadata, "r") as file:
                loaded = yaml.safe_load(file)

            meta_data = HamiltonianData(**loaded)

        else:
            raise FileNotFoundError(f"no file available in {filepath}")
        logger.info(f"loaded checkpoint from {folder}")

        return cls(
            max_l=meta_data.max_l,
            hamiltonian_couplings=hamiltonian_couplings,
            jump_couplings=jump_couplings,
        )

    def save_checkpoint(self, folder: str):
        """!
        save Lindbladian under folder.
        """
        # create state metadata
        meta_data = HamiltonianData(max_l=self.max_l, L=self.L, name=self.name)
        meta_data = asdict(meta_data)

        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)

        hamiltonian_filepath = p / "lindbladian.pkl"
        hamiltonian_metadata = p / "lindbladian_config.yaml"
        if hamiltonian_filepath.is_file() and hamiltonian_metadata.is_file():
            logger.warning("files exist")

        # save as pickle
        with open(hamiltonian_filepath, "wb") as file:
            pickle.dump(
                [self.hamiltonian_couplings, self.jump_couplings],
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # save as yaml
        with open(hamiltonian_metadata, "w") as file:
            yaml.dump(meta_data, file)

        logger.info(f"saved state in {hamiltonian_filepath}")

    def __str__(self):
        string = "Lindbladian:\n\n"

        for key, value in self.__dict__.items():
            if key not in [
                "lindbladian_dict",
                "subsystem_hamiltonian",
                "L_operators",
                "operator",
                "energy_current",
            ]:
                string += "\t{:<25}: {}\n".format(key, value)

        # get smallest and largest key
        for ell in range(self.max_l + 1):
            n_min, n_max = self.subsystem_hamiltonian.boundaries(ell)
            if n_min is None:
                continue
            else:
                string += "\t{:<25}: {}\n".format("defined between", (n_min, n_max))
                break
        list_of_ell_values = []
        for ell in range(self.max_l + 1):
            n_min, n_max = self.subsystem_hamiltonian.boundaries(ell)
            if n_min is None:
                continue
            else:
                list_of_ell_values += [ell]

        string += "\t{:<25}: {}\n".format("levels", list_of_ell_values)
        return string

    def __eq__(self, other):
        if not isinstance(other, Lindbladian):
            return False
        else:
            return (
                self.operator.to_array() == other.operator.to_array()
                and self.max_l == other.max_l
                and self.L_operators.to_array() == other.L_operators.to_array()
            )
