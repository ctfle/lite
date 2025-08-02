import pytest
from scipy.stats import unitary_group
from local_information.typedefs import SystemOperator
from local_information.core.minimization.minimization import *
from local_information.operators.hamiltonian import Hamiltonian
from local_information.lattice.lattice_dict import LatticeDict
from local_information.state.state import State
from local_information.mpi.mpi_funcs import get_mpi_variables
from local_information.mpi.mpi import Distributor

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()

np.random.seed(42)


@pytest.fixture
def test_state_1():
    # asymptotic state
    system = [
        [np.array([[0.2, 0.0], [0.0, 0.8]])],
        [np.array([[0.5, 0.0], [0.0, 0.5]])],
    ]
    return State.build(system, 1)


@pytest.fixture
def test_state_2():
    # state defined for the entire system
    L = 10
    system_matrices = [np.array([[0.4, 0.0], [0.0, 0.6]]) for _ in range(L)]
    system = [system_matrices, []]
    return State.build(system, 1)


@pytest.fixture
def test_state_inf_temp():
    # asymptotic state
    system = [
        [np.array([[0.5, 0.0], [0.0, 0.5]])],
        [np.array([[0.5, 0.0], [0.0, 0.5]])],
    ]
    return State.build(system, 1)


@pytest.fixture(scope="function")
def test_thermal(request):
    max_l = 5
    L = 11
    (J_x, J_y, J_z, h_x, h_y, h_z) = request.param
    J_x_list = [J_x for _ in range(L)]
    J_y_list = [J_x for _ in range(L)]
    J_z_list = [J_x for _ in range(L)]
    hx_list = [h_x for _ in range(L)]
    hy_list = [h_y for _ in range(L)]
    hz_list = [h_z for _ in range(L)]
    hamiltonian_couplings = [
        ["zz", J_z_list],
        ["xx", J_x_list],
        ["yy", J_y_list],
        ["x", hx_list],
        ["y", hy_list],
        ["z", hz_list],
    ]
    # build the Hamiltonian
    hamiltonian = Hamiltonian(max_l, hamiltonian_couplings)
    # build the state using Hamiltonian
    site_0 = np.eye(2) / 2
    site_1 = (
        np.eye(2**3, dtype=np.complex128)
        - 0.1 * hamiltonian.subsystem_hamiltonian[(L // 2, 2)].toarray()
    ) / 8
    system = [[site_1], [site_0]]
    state = State.build(system, 1)
    return state


@pytest.fixture(scope="function")
def random_hamiltonian(request):
    max_l = 5
    L = 20
    (J_x, J_y, J_z, h_x, h_y, h_z) = request.param
    J_x_list = [J_x for _ in range(L)]
    J_y_list = [J_x for _ in range(L)]
    J_z_list = [J_x for _ in range(L)]
    hx_list = [h_x for _ in range(L)]
    hy_list = [h_y for _ in range(L)]
    hz_list = [h_z for _ in range(L)]
    hamiltonian_couplings = [
        ["zz", J_z_list],
        ["xx", J_x_list],
        ["yy", J_y_list],
        ["x", hx_list],
        ["y", hy_list],
        ["z", hz_list],
    ]
    return Hamiltonian(max_l, hamiltonian_couplings)


@pytest.fixture(scope="function")
def random_lattice_dict(request):
    (level, number_of_matrices) = request.param
    density_matrices = []
    keys = []
    for n in range(number_of_matrices):
        random_density_matrix = np.diag(
            np.random.uniform(low=0.01, high=1.0, size=2 ** (level + 1))
        )
        random_density_matrix /= np.trace(random_density_matrix)
        random_unitary = unitary_group.rvs(2 ** (level + 1))
        random_density_matrix = (
            random_unitary.conj().T @ random_density_matrix @ random_unitary
        )
        density_matrices.append(random_density_matrix)
        keys.append((n + 0.5 * level, level))

    return LatticeDict.from_list(keys, density_matrices)


def get_minimizer(hamiltonian: SystemOperator, level: int):
    # we explicitly set max_l in TimeEvolutionConfig just to ensure a different value from min_l
    return InformationMinimizer(
        system_operator=hamiltonian,
        config=TimeEvolutionConfig(min_l=level, max_l=level + 2),
    )


def get_split_up_keys(random_lattice: LatticeDict, level) -> list[list[tuple]]:
    distributor = Distributor(random_lattice, level)
    return distributor._split_keys(number_of_splits=2, shift=0)
