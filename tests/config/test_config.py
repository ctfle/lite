import os
import shutil
from dataclasses import asdict
from local_information.config.config import TimeEvolutionConfig


class TestTimeEvolutionConfig:
    def test_to_yaml_from_yaml(
        self,
    ):
        folder = "test_folder"
        config = TimeEvolutionConfig()
        config.to_yaml(folder)
        # save as yaml
        assert os.path.isdir("test_folder")
        assert os.path.isfile("test_folder/config.yaml")

        # load yaml
        loaded_config = TimeEvolutionConfig.from_yaml(folder)
        for key, value in asdict(loaded_config).items():
            assert value == asdict(config)[key]

        shutil.rmtree("test_folder")

    def test_to_yaml_from_yaml_default(
        self,
    ):
        config = TimeEvolutionConfig()
        config.to_yaml()
        # save as yaml
        assert os.path.isdir("./data")
        assert os.path.isfile("data/config.yaml")

        # load yaml
        loaded_config = TimeEvolutionConfig.from_yaml(directory="data")
        for key, value in asdict(loaded_config).items():
            assert value == asdict(config)[key]

        shutil.rmtree("data")

    def test_corerct_logging_and_checkpoint_folders(
        self,
    ):
        folder = "test_folder"
        config = TimeEvolutionConfig(checkpoint_folder=folder)
        assert config.logging_config.logging_folder == folder + "/logging"
