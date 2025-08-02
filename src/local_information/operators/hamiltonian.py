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
    compute_H_onsite_operator_commutator,
    compute_HH_commutator,
    Operator,
    HamiltonianData,
)

if TYPE_CHECKING:
    from local_information.typedefs import Coupling

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class Hamiltonian(Operator):
    """!
    Hamiltonian organized as LatticeDicts to compute and time evolve subsystem density matrices
    based on the Petz map and the corresponding information lattice and currents. Stores the couplings,
    the `operator` (decomposed into locel terms and stored as LatticeDict) and all the `subsystem_hamiltonian`s.
    Note: there is a difference between `operator` and `subsystem_hamiltonian`. Summation of all values of the
    `operator` LatticeDict yields the Hamitlonian of the entire system. This is not the case for
    `subsystem_hamiltonian` (since subsystems overlap different terms would be counted multiple times).
    """

    def __init__(self, max_l: int, hamiltonian_couplings: Coupling):
        # the 'normalized' Hamiltonian elements weighted with the correct factor
        super().__init__(hamiltonian_couplings, name="Hamiltonian")

        # maximum correlation length. As long as dyn_max_l < max_l, the exact Petz map will be used
        self.max_l = max_l
        self.hamiltonian_couplings = hamiltonian_couplings

        # All relevant subsystem Hamiltonians.
        # Does not result in the Hamiltonian if all values of the corresponding dict are summed up
        self.subsystem_hamiltonian = construct_operator_dict(
            hamiltonian_couplings, self.max_l, self.range_, self.L
        )

    def operator_current(self, operator: Operator) -> dict[tuple[float, int], LatticeDict]:
        """
        Computes the commutator of the Hamiltonian with each element of an onsite (!) operator.
        """
        op_current = dict()
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
        """
        Computes the commutator of the Hamiltonian with each of its decomposed local elements.
        """
        e_current:  dict[tuple[float, int], LatticeDict] = {}
        for key in self.operator.keys_at_level(self.range_):
            e_current[key] = compute_HH_commutator(self.operator, key)

        return e_current

    @classmethod
    def from_checkpoint(cls, folder: str) -> Hamiltonian:
        """
        Load Hamiltonian from checkpoint.
        """
        p = Path(folder)
        hamiltonian_filepath = p / "hamiltonian.pkl"
        hamiltonian_metadata = p / "hamiltonian_config.yaml"

        if hamiltonian_filepath.is_file() and hamiltonian_metadata.is_file():
            # load file
            with open(hamiltonian_filepath, "rb") as file:
                hamiltonian_couplings = pickle.load(file)
            # load meta_data yaml
            with open(hamiltonian_metadata, "r") as file:
                loaded = yaml.safe_load(file)

            meta_data = HamiltonianData(**loaded)

        else:
            raise FileNotFoundError(f"no file available in {hamiltonian_filepath}")
        logger.info(f"loaded checkpoint from {folder}")

        return cls(max_l=meta_data.max_l, hamiltonian_couplings=hamiltonian_couplings)

    def save_checkpoint(self, folder: str):
        """
        Save hamiltonian under folder.
        """
        # create state metadata
        meta_data = HamiltonianData(max_l=self.max_l, L=self.L, name=self.name)
        meta_data = asdict(meta_data)

        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)

        hamiltonian_filepath = p / "hamiltonian.pkl"
        hamiltonian_metadata = p / "hamiltonian_config.yaml"
        if hamiltonian_filepath.is_file() and hamiltonian_metadata.is_file():
            logger.warning("files exist")

        # save as pickle
        with open(hamiltonian_filepath, "wb") as file:
            pickle.dump(
                self.hamiltonian_couplings, file, protocol=pickle.HIGHEST_PROTOCOL
            )

        # save as yaml
        with open(hamiltonian_metadata, "w") as file:
            yaml.dump(meta_data, file)

        logger.info(f"saved state in {hamiltonian_filepath}")
        pass

    def __eq__(self, other):
        if not isinstance(other, Hamiltonian):
            return False
        else:
            return (
                self.operator.to_array() == other.operator.to_array()
                and self.max_l == other.max_l
            )

    def __str__(self):
        string = "Hamiltonian:\n\n"

        for key, value in self.__dict__.items():
            if key not in [
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
