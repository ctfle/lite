import os
import sys

os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import pickle
import src as li
from local_information.mpi.mpi_funcs import get_mpi_variables

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


def main():
    min_l = 3
    max_l = 6
    L = 101
    dissipation_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dissipation_strength = dissipation_list[1]

    J_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    J = J_list[1]
    hz = 0.2

    J_list = [J for j in range(L)]
    z_list = [hz for j in range(L)]
    hamiltonian_couplings = [["xx", J_list], ["yy", J_list], ["z", z_list]]

    # lindblad part for the evolution
    z_list = [dissipation_strength for j in range(L)]
    jump_couplings = [["z", z_list]]  # [['+',plus_list],['-',minus_list], ['z',z_list]]

    # lindbaldians
    lindbladian_Hdd = li.src.operators.Lindbladian(
        max_l, hamiltonian_couplings, jump_couplings
    )

    # x-magnetisation
    z_ = [1.0 for j in range(L)]
    z_list = [["z", z_]]
    z_mag = li.src.operators.Operator(z_list)

    # observables
    z_operator_current = lindbladian_Hdd.operator_current(z_mag)
    print(z_mag)


if __name__ == "__main__":
    main()
