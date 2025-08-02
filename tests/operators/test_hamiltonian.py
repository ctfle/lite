import os
import shutil

import numpy as np
import pytest

from local_information.operators.hamiltonian import Hamiltonian
from local_information.operators.operator import add_spins


class TestHamiltonian:
    @pytest.fixture
    def test_z_hamiltonian(self):
        L = 10
        J = 0.25
        hL = 0.125
        J_list = [J for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [["zz", J_list], ["z", hL_list]]
        return Hamiltonian(5, hamiltonian_couplings)

    @pytest.fixture
    def test_x_hamiltonian(self):
        L = 2
        J = 0.25
        hL = 0.125
        J_list = [J for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [["zz", J_list], ["z", hL_list]]
        return Hamiltonian(2, hamiltonian_couplings)

    @pytest.fixture
    def test_z_x_hamiltonian(self):
        L = 10
        hT_list = [1.0 for _ in range(L)]
        hamiltonian_couplings = [["zz", hT_list], ["x", hT_list]]
        return Hamiltonian(5, hamiltonian_couplings)

    @pytest.fixture
    def test_mixed_field_ising_hamiltonian(self):
        L = 10
        J_list = [1.0 for _ in range(L)]
        hT_list = [0.9045 for _ in range(L)]
        hL_list = [0.805 for _ in range(L)]
        hamiltonian_couplings = [["zz", J_list], ["x", hT_list], ["z", hL_list]]
        return Hamiltonian(5, hamiltonian_couplings)

    def test_energy_current_basics(self, test_x_hamiltonian, test_z_hamiltonian):
        energy_current = test_x_hamiltonian.energy_current
        for key, val_dict in energy_current.items():
            print(val_dict)
            for _, val in val_dict.items():
                assert np.allclose(np.linalg.norm(val), 0.0)

        energy_current = test_z_hamiltonian.energy_current
        for key, val_dict in energy_current.items():
            for _, val in val_dict.items():
                assert np.allclose(np.linalg.norm(val), 0.0)
        pass

    def test_energy_current_advanced(self, test_z_x_hamiltonian):
        energy_current = test_z_x_hamiltonian.energy_current
        # energy_current is dict since its values are LatticeDict objects
        # get smallest and largest keys
        n_min = 10
        n_max = 0
        for key in energy_current.keys():
            if key[0] <= n_min:
                n_min = key[0]
            if key[0] >= n_max:
                n_max = key[0]

        for key, val_dict in energy_current.items():
            assert len(val_dict.keys()) <= 2
            if key[0] == n_min or key[0] == n_max:
                assert len(val_dict.keys()) == 1
            else:
                assert len(val_dict.keys()) == 2

        # compute the commutator
        # all local Hamiltonians are the same here, so we only need to compute
        # a single commutator. First get the corresponding operator
        # for any of the keys
        operator = test_z_x_hamiltonian.operator[(n_min, 1)]
        # construct two Hamitlonians by adding a single spin right and left
        operator_left = add_spins(operator, 1, orientation="left")
        operator_right = add_spins(operator, 1, orientation="right")
        # current towards the right
        commutator_right = (
            operator_left @ operator_right - operator_right @ operator_left
        )
        # current towards the left
        commutator_left = (
            operator_right @ operator_left - operator_left @ operator_right
        )
        # compare all entries in the tests hamiltonians to the two commutator
        for key, val_dict in energy_current.items():
            for val in val_dict.values():
                assert np.allclose(val, commutator_right) or np.allclose(
                    val, commutator_left
                )
                if key[0] == n_min:
                    assert np.allclose(val, commutator_right)
                elif key[0] == n_max:
                    assert np.allclose(val, commutator_left)
        pass

    def test_energy_current(self, test_mixed_field_ising_hamiltonian):
        energy_current = test_mixed_field_ising_hamiltonian.energy_current
        # energy_current is dict since its values are LatticeDict objects
        # get smallest and largest keys
        n_min = 10
        n_max = 0
        for key in energy_current.keys():
            if key[0] <= n_min:
                n_min = key[0]
            if key[0] >= n_max:
                n_max = key[0]

        for key, val_dict in energy_current.items():
            assert len(val_dict.keys()) <= 2
            if key[0] == n_min or key[0] == n_max:
                assert len(val_dict.keys()) == 1
            else:
                assert len(val_dict.keys()) == 2

        # same as above
        operator = test_mixed_field_ising_hamiltonian.operator[(n_min, 1)]
        # construct two Hamiltonians by adding a single spin right and left
        operator_left = add_spins(operator, 1, orientation="left")
        operator_right = add_spins(operator, 1, orientation="right")
        # current towards the right
        commutator_right = (
            operator_left @ operator_right - operator_right @ operator_left
        )
        # current towards the left
        commutator_left = (
            operator_right @ operator_left - operator_left @ operator_right
        )
        # compare all entries in the tests hamiltonians to the two commutator
        for key, val_dict in energy_current.items():
            for val in val_dict.values():
                assert np.allclose(val, commutator_right) or np.allclose(
                    val, commutator_left
                )
                if key[0] == n_min:
                    assert np.allclose(val, commutator_right)
                elif key[0] == n_max:
                    assert np.allclose(val, commutator_left)
        pass

    def test_save_checkpoint_from_checkpoint(
        self,
        test_x_hamiltonian,
        test_z_x_hamiltonian,
        test_mixed_field_ising_hamiltonian,
    ):
        folder = "test_folder"
        for test_hamiltonian in [
            test_x_hamiltonian,
            test_z_x_hamiltonian,
            test_mixed_field_ising_hamiltonian,
        ]:
            test_hamiltonian.save_checkpoint(folder)
            # save checkpoint
            assert os.path.isdir("test_folder")
            assert os.path.isfile("test_folder/hamiltonian.pkl")
            assert os.path.isfile("test_folder/hamiltonian_config.yaml")

            # load checkpoint
            loaded_hamiltonian = Hamiltonian.from_checkpoint(folder)
            assert loaded_hamiltonian == test_hamiltonian
            assert loaded_hamiltonian.max_l == test_hamiltonian.max_l
            shutil.rmtree("test_folder")
        pass
