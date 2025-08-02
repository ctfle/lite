from __future__ import annotations

import logging
import os.path
import pickle
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, List, AnyStr, Dict, Any
from typing import Union

import yaml
from attrs import define
from cattrs import structure, unstructure, Converter
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override

from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables

from local_information.typedefs import SystemOperator
from local_information.operators.observables import (
    diff_const,
    diff_length,
    energy_distribution,
)
from local_information.state.state import State

logger = logging.getLogger()

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


@define
class DataConfig:
    observables: List = []
    info_lattice: bool = False
    info_current: bool = False
    diffusion_length: bool = True
    diffusion_const: bool = True
    energy_distribution: bool = False
    times: bool = True
    system_size: bool = True

    def to_dict(self):
        if self.observables:
            converter = Converter()
            ust_hook = make_dict_unstructure_fn(
                DataConfig,
                converter,
                observables=override(unstruct_hook=lambda v: [i.__name__ for i in v]),
            )
            converter.register_unstructure_hook(DataConfig, ust_hook)
            return converter.unstructure(self)
        else:
            return unstructure(self)

    def to_yaml(self, folder):
        """save config as yaml"""
        if not os.path.exists(folder):
            os.makedirs(folder)

        p = Path(folder)
        config_path = p / "data_config.yaml"
        dict_repr = self.to_dict()
        with open(config_path, "w") as file:
            yaml.dump(dict_repr, file)
        pass

    @classmethod
    def from_yaml(cls, folder: str, full_module_path: str = "") -> DataConfig:
        """initialize config from yaml"""
        p = Path(folder)
        config_path = p / "data_config.yaml"

        if config_path.is_file():
            # load yaml
            with open(config_path, "r") as file:
                loaded_dict = yaml.safe_load(file)

        # setup observables if needed
        if loaded_dict["observables"]:
            # parse strings to function
            folder_path, module = full_module_path.rsplit("/", maxsplit=1)
            sys.path.insert(1, folder_path)
            module_name, _ = module.rsplit(".", maxsplit=1)

            m = import_module(module_name)
            # parse observables from str to callable
            converter = Converter()
            st_hook = make_dict_structure_fn(
                DataConfig,
                converter,
                observables=override(
                    struct_hook=lambda v, _: [getattr(m, i) for i in v]
                ),
            )
            converter.register_structure_hook(DataConfig, st_hook)

            logger.info(f"loaded DataConfig from {folder}")
            return converter.structure(loaded_dict, DataConfig)
        else:
            return structure(loaded_dict, cls)


@dataclass
class DataContainer:
    config: DataConfig = DataConfig()

    def __post_init__(self):
        self.get_default_observables = DefaultObservables()
        # hand over all default observables
        self._get_empty_container()

    def _get_empty_container(self):
        self.default_observables_dict: Dict[AnyStr, list] = (
            self.set_default_observables_dict()
        )
        # hand over all custom observables
        if self.config.observables:
            self.custom_observables_dict: Dict[AnyStr, list] = dict()
            for observable in self.config.observables:
                self.custom_observables_dict[observable.__name__] = []
        else:
            self.custom_observables_dict = None

    @classmethod
    def from_yaml(cls, folder: str, full_module_path: str = "") -> DataContainer:
        config = DataConfig.from_yaml(folder, full_module_path=full_module_path)
        return cls(config=config)

    def to_yaml(self, folder: str):
        """save config as yaml"""
        self.config.to_yaml(folder)

    def set_default_observables_dict(self) -> dict:
        default_obs_dict = dict()
        for field_name, value in unstructure(self.config).items():
            if field_name == "observables":
                continue
            if getattr(self.config, field_name):
                default_obs_dict[field_name] = []
        return default_obs_dict

    def update_default_observables(
        self,
        density_matrix: LatticeDict,
        information_dict: LatticeDict,
        state: State,
        operator: SystemOperator,
    ):
        for observable_name, data_list in self.default_observables_dict.items():
            # call the helper class to compute all default observables
            observable_result = self.get_default_observables(
                observable=observable_name,
                densitz_matrix=density_matrix,
                information_dict=information_dict,
                state=state,
                operator=operator,
            )
            if observable_result is not None:
                data_list.append(observable_result)

    def update_custom_observables(self, rho_dict: LatticeDict):
        """
        All worker processes should enter the computation of custom observables since
        we do not specify the exact functions to be called (mpi could be used).
        """
        for observable in self.config.observables:
            observable_result = observable(rho_dict)
            self.custom_observables_dict[observable.__name__].append(observable_result)

    def save_checkpoint(self, folder: str, warning: bool = True):
        """saving and loading should only take place at root"""
        if RANK == 0:
            path = Path(folder)
            path.mkdir(parents=True, exist_ok=True)

            # save default data
            self._save_default_observables(path=path, warning=warning)

            # save custom observables
            self._save_custom_observables(path=path, warning=warning)
            logger.info(f"saved data under {path}")

            # save config values in yaml
            self.config.to_yaml(folder)
        else:
            # empty the buffer for all other MPI processes
            self._get_empty_container()

    def _save_default_observables(self, path: Path, warning: bool = True):
        # save default data
        for observable_name, data_list in self.default_observables_dict.items():
            filename = observable_name + ".pkl"
            filepath = path / filename
            if filepath.is_file() and warning:
                logger.warning(f"{filepath} exists and is overwritten")
            with open(filepath, "wb") as file:
                pickle.dump(data_list, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_custom_observables(self, path: Path, warning: bool = True):
        if self.custom_observables_dict is not None:
            for observable_name, data_list in self.custom_observables_dict.items():
                filename = observable_name + ".pkl"
                filepath = path / filename
                if filepath.is_file() and warning:
                    logger.warning(f"{filepath} exists and is overwritten")
                with open(filepath, "wb") as file:
                    pickle.dump(data_list, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, folder: str):
        """load data from checkpoint at root"""
        p = Path(folder)
        if RANK == 0:
            if self.filepath_exists(folder):
                for observable_name, _ in self.default_observables_dict.items():
                    filename = observable_name + ".pkl"
                    filepath = p / filename

                    with open(filepath, "rb") as file:
                        loaded_data = pickle.load(file)
                    self.default_observables_dict[observable_name] = loaded_data

                if self.custom_observables_dict is not None:
                    for observable_name, _ in self.custom_observables_dict.items():
                        filename = observable_name + ".pkl"
                        filepath = p / filename

                        with open(filepath, "rb") as file:
                            loaded_data = pickle.load(file)
                        self.custom_observables_dict[observable_name] = loaded_data

    @staticmethod
    def load_times(folder: str) -> list:
        p = Path(folder)
        filename = "times.pkl"
        filepath = p / filename
        with open(filepath, "rb") as file:
            times = pickle.load(file)
        return times

    def attach_to_existing_file(self, folder: str):
        """loads data into buffer, attaches data and saves_checkpoint"""
        if RANK == 0:
            p = Path(folder)
            if self.filepath_exists(folder):
                for observable_name, data_list in self.default_observables_dict.items():
                    filename = observable_name + ".pkl"
                    filepath = p / filename
                    with open(filepath, "rb") as file:
                        loaded_data = pickle.load(file)

                    self.default_observables_dict[observable_name] = (
                        loaded_data + data_list
                    )

                if self.custom_observables_dict is not None:
                    for (
                        observable_name,
                        data_list,
                    ) in self.custom_observables_dict.items():
                        filename = observable_name + ".pkl"
                        filepath = p / filename
                        with open(filepath, "rb") as file:
                            loaded_data = pickle.load(file)
                            self.custom_observables_dict[observable_name] = (
                                loaded_data + data_list
                            )

                self.save_checkpoint(folder, warning=False)
                logger.info(f"attached data to existing files in {folder}")
            else:
                self.save_checkpoint(folder, warning=True)
                logger.info(f"saved data in {folder}")
        else:
            # empty the buffer for all other MPI processes
            self._get_empty_container()

    def filepath_exists(self, folder: str) -> bool:
        """checks if files exist"""
        exists = True
        p = Path(folder)
        for observable_name, _ in self.default_observables_dict.items():
            filename = observable_name + ".pkl"
            filepath = p / filename
            if not filepath.is_file():
                exists = False
        return exists

    def return_latest(self) -> dict:
        """Return last updated values."""
        latest_values = dict()
        for observable_name, observable_values in self.custom_observables_dict.items():
            if len(observable_values) == 0:
                latest_values[observable_name] = None
            else:
                latest_values[observable_name] = observable_values[-1]
        for observable_name, observable_values in self.default_observables_dict.items():
            if len(observable_values) == 0:
                latest_values[observable_name] = None
            else:
                latest_values[observable_name] = observable_values[-1]
        return latest_values


class DefaultObservables:
    """Helper class to compute all default observables"""

    state: State = None
    operator: SystemOperator = None
    density_matrix: LatticeDict = None

    def __init__(self):
        self.observables = {
            "diffusion_const": self.get_diffusion_const,
            "diffusion_length": self.get_diffusion_length,
            "energy_distribution": self.get_energy_distribution,
            "times": self.get_time,
            "system_size": self.get_system_size,
            "info_lattice": self.get_information_lattice,
            "info_current": self.get_information_current,
        }

    def __call__(
        self,
        observable: str,
        densitz_matrix: LatticeDict,
        information_dict: LatticeDict,
        state: State,
        operator: SystemOperator,
    ) -> Union[None, Any]:
        self.state = state
        self.operator = operator
        self.density_matrix = densitz_matrix
        self.information_dict = information_dict

        if observable in self.observables and RANK == 0:
            observable_result = self.compute_observable(observable)
        else:
            observable_result = None

        return observable_result

    def compute_observable(self, observable: str) -> Any:
        return self.observables[observable]()

    def get_diffusion_const(self):
        return diff_const(
            density_matrix=self.density_matrix, state=self.state, operator=self.operator
        )

    def get_diffusion_length(self):
        return diff_length(density_matrix=self.density_matrix, operator=self.operator)

    def get_energy_distribution(self):
        return energy_distribution(density_matrix=self.density_matrix, operator=self.operator)

    def get_system_size(self):
        return self.state.system_size

    def get_time(self):
        return self.state.current_time

    def get_information_lattice(self):
        return self.information_dict

    def get_information_current(self):
        return self.state.get_information_current(self.operator)
