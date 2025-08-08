import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from mock import MagicMock
from unittest.mock import patch

from local_information.operators.hamiltonian import Hamiltonian
from local_information.state.state import State
from local_information.config.monitor import DataConfig, DataContainer
from local_information.lattice.lattice_dict import LatticeDict
from local_information.config.monitor import DefaultObservables


@pytest.fixture(scope="function")
def mock_func(request):
    """mock a function to return values"""
    value = request.param
    mock = MagicMock()
    mock.return_value = value
    return mock


def some_test_func(rho):
    return 1


def some_other_test_func(rho):
    return 1


class TestDataConfig:
    @pytest.fixture
    def test_data_config_observables(self):
        data_config = DataConfig(observables=[some_test_func, some_other_test_func])
        return data_config

    @pytest.fixture
    def test_data_config_no_observables(self):
        data_config = DataConfig()
        return data_config

    def test_to_yaml_from_yaml(self, test_data_config_no_observables):
        folder = "test_folder"
        test_data_config_no_observables.to_yaml(folder)
        # save as yaml
        assert os.path.isdir("test_folder")
        assert os.path.isfile("test_folder/data_config.yaml")

        # load yaml
        loaded_data_config = DataConfig.from_yaml(folder)
        test_data_config_dict = test_data_config_no_observables.to_dict()
        for key, value in loaded_data_config.to_dict().items():
            assert value == test_data_config_dict[key]

        shutil.rmtree("test_folder")

    def test_to_yaml_from_yaml_observables(self, test_data_config_observables):
        folder = "test_folder"
        test_data_config_observables.to_yaml(folder)
        # save as yaml
        assert os.path.isdir("test_folder")
        assert os.path.isfile("test_folder/data_config.yaml")

        # load yaml
        working_dir = Path(__file__).parent.resolve().as_posix()
        # load the observables some_test_func and some_other_test_func from the current file
        loaded_data_config = DataConfig.from_yaml(
            folder, full_module_path=working_dir + "/test_monitor.py"
        )
        test_data_config_dict = test_data_config_observables.to_dict()
        for key, value in loaded_data_config.to_dict().items():
            assert value == test_data_config_dict[key]

        shutil.rmtree("test_folder")


class TestData:
    @pytest.fixture
    def test_state(self):
        # some initial state
        system = [
            [np.array([[0.4, 0.0], [0.0, 0.6]])],
            [np.array([[0.5, 0.0], [0.0, 0.5]])],
        ]
        return State.build(system, 1)

    @pytest.fixture
    def test_hamiltonian(self):
        # hamiltonian
        max_l = 5
        L = 10
        J = 0.25
        hL = 0.125
        hT = -0.2625
        J_list = [J for _ in range(L)]
        hT_list = [hT for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        hamiltonian_couplings = [["zz", J_list], ["z", hL_list], ["x", hT_list]]
        return Hamiltonian(max_l, hamiltonian_couplings)

    @pytest.fixture
    def test_data(self):
        return DataContainer(config=DataConfig())

    def test_to_yaml_from_yaml(self, test_data):
        folder = "test_folder"
        test_data.to_yaml(folder)
        assert os.path.isdir("test_folder")
        assert os.path.isfile("test_folder/data_config.yaml")

        # load yaml
        loaded_data = DataContainer.from_yaml(folder)
        test_data_config_dict = test_data.config.to_dict()
        for key, value in loaded_data.config.to_dict().items():
            assert value == test_data_config_dict[key]

        shutil.rmtree("test_folder")

    @pytest.mark.parametrize(
        "data, default_dict_length, default_dict_keys",
        [
            (
                DataContainer(config=DataConfig()),
                4,
                ["diffusion_length", "diffusion_const", "system_size", "times"],
            ),
            (
                DataContainer(config=DataConfig(diffusion_length=False)),
                3,
                ["diffusion_const", "system_size", "times"],
            ),
            (
                DataContainer(
                    config=DataConfig(diffusion_length=False, diffusion_const=False)
                ),
                2,
                ["system_size", "times"],
            ),
        ],
    )
    def test_default_observables_dict(
        self, data, default_dict_length, default_dict_keys
    ):
        # check if the default observables are initialized correctly
        for key, value in data.default_observables_dict.items():
            assert value == []
            assert key in default_dict_keys
        assert len(data.default_observables_dict) == default_dict_length
        assert data.custom_observables_dict is None

    @pytest.mark.parametrize(
        "data, custom_dict_length, custom_dict_keys",
        [
            (
                DataContainer(
                    config=DataConfig(
                        observables=[some_test_func, some_other_test_func]
                    )
                ),
                2,
                ["some_test_func", "some_other_test_func"],
            ),
            (
                DataContainer(config=DataConfig(observables=[some_test_func])),
                1,
                ["some_test_func"],
            ),
        ],
    )
    def test_custom_observables_dict(self, data, custom_dict_length, custom_dict_keys):
        for key, value in data.custom_observables_dict.items():
            assert value == []
            assert key in custom_dict_keys
        assert len(data.custom_observables_dict) == custom_dict_length

    @pytest.mark.parametrize(
        "mock_func, value",
        [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        indirect=["mock_func"],
    )
    def test_update_custom_observable_dict(self, mock_func, value):
        def some_func(rho):
            return mock_func(rho)

        data = DataContainer(config=DataConfig(observables=[some_func]))
        test_rho_dict = LatticeDict()

        for i in range(10):
            data.update_custom_observables(test_rho_dict)
        assert len(data.custom_observables_dict["some_func"]) == 10
        for i in range(10):
            assert data.custom_observables_dict["some_func"][i] == value

    def test_save_checkpoint_load_checkpoint(self):
        test_rho_dict = LatticeDict()
        data = DataContainer(config=DataConfig(observables=[some_test_func]))
        for i in range(10):
            data.update_custom_observables(test_rho_dict)

        # save checkpoint
        folder = "test_folder"
        data.save_checkpoint(folder)

        assert os.path.isdir("test_folder")
        assert os.path.isfile("test_folder/some_test_func.pkl")
        assert os.path.isfile("test_folder/diffusion_length.pkl")
        assert os.path.isfile("test_folder/diffusion_const.pkl")
        assert os.path.isfile("test_folder/system_size.pkl")
        assert os.path.isfile("test_folder/times.pkl")
        assert os.path.isfile("test_folder/data_config.yaml")

        # load checkpoint
        loaded_data = DataContainer(config=DataConfig(observables=[some_test_func]))
        loaded_data.load_checkpoint(folder=folder)
        for key, value in loaded_data.custom_observables_dict.items():
            assert key == "some_test_func"
            assert len(value) == 10
            for item in value:
                assert item == some_test_func(test_rho_dict)

        shutil.rmtree("test_folder")

    def test_update_default_observables_dict_with_times_and_system_size(
        self, test_state, test_hamiltonian
    ):
        # data object having only times and system_size
        data = DataContainer(
            config=DataConfig(diffusion_const=False, diffusion_length=False)
        )
        test_rho_dict = LatticeDict()

        some_times = [1, 2, 3, 4]
        some_system_sizes = [10, 11, 12, 13]
        for test_time, test_system_size in zip(some_times, some_system_sizes):
            test_state.system_size = test_system_size
            test_state.current_time = test_time
            data.update_default_observables(
                density_matrix=test_rho_dict,
                information_dict=LatticeDict(),
                operator=test_hamiltonian,
                state=test_state,
            )

        assert data.default_observables_dict["times"] == some_times
        assert data.default_observables_dict["system_size"] == some_system_sizes

    @patch.object(DefaultObservables, "get_diffusion_length")
    @patch.object(DefaultObservables, "get_diffusion_const")
    def test_update_default_observables_dict_with_diffusion_const_and_length(
        self,
        get_diffusion_const,
        get_diffusion_length,
        test_state,
        test_hamiltonian,
    ):
        data = DataContainer(
            config=DataConfig(diffusion_const=True, diffusion_length=True)
        )
        test_rho_dict = LatticeDict()
        diff_constants = [0.1, 0.2, 0.3, 0.4]
        diff_lengths = [1.1, 1.2, 1.3, 1.4]
        for diff_constant, diff_length in zip(diff_constants, diff_lengths):
            get_diffusion_length.return_value = diff_length
            get_diffusion_const.return_value = diff_constant
            data.update_default_observables(
                density_matrix=test_rho_dict,
                information_dict=LatticeDict(),
                operator=test_hamiltonian,
                state=test_state,
            )
        assert data.default_observables_dict["diffusion_const"] == diff_constants
        assert data.default_observables_dict["diffusion_length"] == diff_lengths

    @patch.object(DefaultObservables, "get_diffusion_length")
    @patch.object(DefaultObservables, "get_diffusion_const")
    def test_update_default_observables_dict_with_diffusion_diffusion_const(
        self,
        get_diffusion_const,
        get_diffusion_length,
        test_state,
        test_hamiltonian,
    ):
        data = DataContainer(
            config=DataConfig(diffusion_const=True, diffusion_length=False)
        )
        test_rho_dict = LatticeDict()
        diff_constants = [0.1, 0.2, 0.3, 0.4]
        diff_lengths = [1.1, 1.2, 1.3, 1.4]
        for diff_constant, diff_length in zip(diff_constants, diff_lengths):
            get_diffusion_length.return_value = diff_length
            get_diffusion_const.return_value = diff_constant
            data.update_default_observables(
                density_matrix=test_rho_dict,
                information_dict=LatticeDict(),
                operator=test_hamiltonian,
                state=test_state,
            )
        assert data.default_observables_dict["diffusion_const"] == diff_constants
        with pytest.raises(KeyError):
            _ = data.default_observables_dict["diffusion_length"]

    @patch.object(DefaultObservables, "get_information_lattice")
    def test_update_default_observables_dict_with_diffusion_const_and_length_2(
        self,
        get_information_lattice,
        test_state,
        test_hamiltonian,
    ):
        data = DataContainer(
            config=DataConfig(
                diffusion_const=False, diffusion_length=False, info_lattice=True
            )
        )
        test_rho_dict = LatticeDict()
        for i in range(10):
            keys = [(1, 2), (3, 4), (4, 5), (6, 7)]
            values = [1.1 + i, 1.2 + i, 1.3 + i, 1.4 + i]
            info_lattice = LatticeDict(zip(keys, values))
            get_information_lattice.return_value = info_lattice
            data.update_default_observables(
                density_matrix=test_rho_dict,
                information_dict=LatticeDict(),
                operator=test_hamiltonian,
                state=test_state,
            )
            assert data.default_observables_dict["info_lattice"][-1] == info_lattice

    @pytest.mark.parametrize(
        "mock_func, value",
        [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        indirect=["mock_func"],
    )
    def test_attach_to_existing_file(self, mock_func, value):
        def some_func(rho):
            return mock_func(rho)

        folder = "test_folder"
        test_rho_dict = LatticeDict()

        data = DataContainer(config=DataConfig(observables=[some_func]))
        data.save_checkpoint(folder)
        for i in range(10):
            data = DataContainer(config=DataConfig(observables=[some_func]))
            data.update_custom_observables(test_rho_dict)
            # attach files
            data.attach_to_existing_file(folder)

        # load the checkpoint
        data = DataContainer(config=DataConfig(observables=[some_func]))
        data.load_checkpoint(folder)
        assert len(data.custom_observables_dict["some_func"]) == 10
        for i in range(10):
            assert data.custom_observables_dict["some_func"][i] == value

        shutil.rmtree("test_folder")

    @pytest.mark.parametrize(
        "mock_func, last_val",
        [
            (7, 7),
            (7.456, 7.456),
        ],
        indirect=["mock_func"],
    )
    def test_return_latest(self, mock_func, last_val):
        def some_func(rho):
            return mock_func(rho)

        test_rho_dict = LatticeDict()
        data = DataContainer(config=DataConfig(observables=[some_func]))
        data.update_custom_observables(test_rho_dict)

        value_dict = data.return_latest()
        for key, value in value_dict.items():
            if key == "some_func":
                assert value == last_val
            else:
                assert value is None
