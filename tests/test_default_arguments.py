import numpy as np
import pytest

from local_information.config import TimeEvolutionConfig
from local_information.config.monitor import DataConfig, DataContainer
from local_information.lindblad_evolution import OpenSystem
from local_information.operators.hamiltonian import Hamiltonian
from local_information.operators.lindbladian import Lindbladian
from local_information.state.state import State
from local_information.time_evolution import ClosedSystem


@pytest.fixture
def state():
    system = [
        [np.array([[0.2, 0.0], [0.0, 0.8]])],
        [np.array([[0.5, 0.0], [0.0, 0.5]])],
    ]
    return State.build(system, 1)


@pytest.fixture
def hamiltonian():
    couplings = [["zz", [1.0] * 5], ["x", [0.5] * 5]]
    return Hamiltonian(2, couplings)


@pytest.fixture
def lindbladian():
    return Lindbladian(2, [["zz", [1.0] * 5], ["x", [0.5] * 5]], [])


def test_closed_system_default_inputs_are_instances(state, hamiltonian):
    system = ClosedSystem(init_state=state, hamiltonian=hamiltonian)
    assert isinstance(system.config, TimeEvolutionConfig)
    assert isinstance(system.data, DataContainer)


def test_closed_system_defaults_are_not_shared(state, hamiltonian):
    first = ClosedSystem(init_state=state, hamiltonian=hamiltonian)
    second = ClosedSystem(init_state=state, hamiltonian=hamiltonian)
    assert first.config is not second.config
    assert first.data is not second.data


def test_open_system_default_inputs_are_instances(state, lindbladian):
    system = OpenSystem(init_state=state, lindbladian=lindbladian)
    assert isinstance(system.config, TimeEvolutionConfig)
    assert isinstance(system.data, DataContainer)


def test_open_system_defaults_are_not_shared(state, lindbladian):
    first = OpenSystem(init_state=state, lindbladian=lindbladian)
    second = OpenSystem(init_state=state, lindbladian=lindbladian)
    assert first.config is not second.config
    assert first.data is not second.data


def test_data_container_default_config_is_not_shared():
    assert DataContainer().config is not DataContainer().config
    assert isinstance(DataContainer().config, DataConfig)
