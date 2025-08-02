import os
import shutil

import pytest

from local_information.operators.lindbladian import Lindbladian


class TestLindbladian:
    @pytest.fixture
    def test_lindbladian_1(self):
        L = 10
        J = 0.25
        hL = 0.125
        hT = -0.2625
        J_list = [J for _ in range(L)]
        hT_list = [hT for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [["zz", J_list], ["z", hL_list], ["x", hT_list]]
        plus_list = [1.0 if j == 5 else 0.0 for j in range(L)]
        minus_list = [0.1 if j == 5 else 0.0 for j in range(L)]
        jump_couplings = [["+", plus_list], ["-", minus_list]]
        return Lindbladian(3, hamiltonian_couplings, jump_couplings)

    @pytest.fixture
    def test_lindbladian_2(self):
        L = 10
        J = 1.0
        hL = 0.2345
        hT = -0.543
        J_list = [J for _ in range(L)]
        hT_list = [hT for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [
            ["xx", J_list],
            ["xx", J_list],
            ["y", hL_list],
            ["x", hT_list],
        ]
        plus_list = [1.0 if j == 5 else 0.0 for j in range(L)]
        minus_list = [0.1 if j == 5 else 0.0 for j in range(L)]
        z_list = [1.0 for _ in range(L)]
        jump_couplings = [["+", plus_list], ["-", minus_list], ["x", z_list]]
        return Lindbladian(3, hamiltonian_couplings, jump_couplings)

    def test_save_checkpoint_from_checkpoint(
        self, test_lindbladian_1, test_lindbladian_2
    ):
        folder = "test_folder"
        for test_lindbladian in [test_lindbladian_1, test_lindbladian_2]:
            test_lindbladian.save_checkpoint(folder)
            # save checkpoint
            assert os.path.isdir("test_folder")
            assert os.path.isfile("test_folder/lindbladian.pkl")
            assert os.path.isfile("test_folder/lindbladian_config.yaml")

            # load checkpoint
            loaded_hamiltonian = Lindbladian.from_checkpoint(folder)
            assert loaded_hamiltonian == test_lindbladian
            assert loaded_hamiltonian.max_l == test_lindbladian.max_l
            shutil.rmtree("test_folder")
        pass
