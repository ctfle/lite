import pytest

from local_information.core.petz_map import *
from copy import deepcopy


class TestPetzMap:
    @pytest.mark.parametrize(
        "A, B, density_matrix_1, density_matrix_2, petz_map_result",
        [
            (
                (1, 0),
                (2, 0),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.eye(4) / 4,
            ),
            (
                (1, 0),
                (2, 0),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ),
            (
                (1, 0),
                (2, 0),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array(
                    [
                        [0.81, 0.0, 0.0, 0.0],
                        [0.0, 0.09, 0.0, 0.0],
                        [0.0, 0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
            ),
            (
                (1, 0),
                (3, 0),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array(
                    [
                        [0.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.81, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.09, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.09, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
                    ]
                ),
            ),
        ],
    )
    def test_petz_map_sqrt_no_overlap(
        self,
        A,
        B,
        density_matrix_1: np.ndarray,
        density_matrix_2: np.ndarray,
        petz_map_result,
    ):
        """tests the petz map using sqrt version if there is no overlap"""
        precomp_dens_A = np_sqrt(density_matrix_1)
        precomp_dens_B = np_sqrt(density_matrix_2)

        petz_map = PetzMap(
            key_A=A,
            key_B=B,
            density_matrix_A=density_matrix_1,
            density_matrix_B=density_matrix_2,
            precomputed_sqrt_or_log_of_density_matrix_A=precomp_dens_A,
            precomputed_sqrt_or_log_of_density_matrix_B=precomp_dens_B,
            sqrt_method=True,
        )
        rho_AB = petz_map.get_combined_system()
        new_key = petz_map.get_new_key()

        assert np.allclose(rho_AB, petz_map_result)
        assert new_key.n == (A[0] + B[0]) / 2
        assert new_key.level == A[1] + abs(B[0] - A[0])

    @pytest.mark.parametrize(
        "A, B, density_matrix_1, density_matrix_2",
        [
            (
                (1, 1),
                (2, 1),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
            ),
            (
                (1, 5),
                (2, 3),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
            ),
        ],
    )
    def test_petz_map_wrong_input(
        self, A, B, density_matrix_1: np.ndarray, density_matrix_2: np.ndarray
    ):
        precomp_dens_A = np_sqrt(density_matrix_1)
        precomp_dens_B = np_sqrt(density_matrix_2)
        with pytest.raises(ValueError):
            p_map = PetzMap(
                key_A=A,
                key_B=B,
                density_matrix_A=density_matrix_1,
                density_matrix_B=density_matrix_2,
                precomputed_sqrt_or_log_of_density_matrix_A=precomp_dens_A,
                precomputed_sqrt_or_log_of_density_matrix_B=precomp_dens_B,
                sqrt_method=True,
            )
            _ = p_map.get_combined_system()

    @pytest.mark.parametrize(
        "A, B, density_matrix_1, density_matrix_2",
        [
            (
                (1, 0),
                (2, 1),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.array(
                    [
                        [0.81, 0.0, 0.0, 0.0],
                        [0.0, 0.09, 0.0, 0.0],
                        [0.0, 0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
            ),
            (
                (1.5, 1),
                (2.5, 1),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
            ),
        ],
    )
    def test_petz_map_wrong_shape(
        self, A, B, density_matrix_1: np.ndarray, density_matrix_2: np.ndarray
    ):
        precomp_dens_A = np_sqrt(density_matrix_1)
        precomp_dens_B = np_sqrt(density_matrix_2)
        with pytest.raises(ValueError):
            p_map = PetzMap(
                key_A=A,
                key_B=B,
                density_matrix_A=density_matrix_1,
                density_matrix_B=density_matrix_2,
                precomputed_sqrt_or_log_of_density_matrix_A=precomp_dens_A,
                precomputed_sqrt_or_log_of_density_matrix_B=precomp_dens_B,
                sqrt_method=True,
            )
            _ = p_map.get_combined_system()

    @pytest.mark.parametrize(
        "A, B, density_matrix_1, density_matrix_2, petz_map_result",
        [
            (
                (1, 0),
                (2, 0),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.eye(4) / 4,
            ),
            (
                (1, 0),
                (2, 0),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array(
                    [
                        [0.81, 0.0, 0.0, 0.0],
                        [0.0, 0.09, 0.0, 0.0],
                        [0.0, 0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
            ),
        ],
    )
    def test_petz_map_log_no_overlap(
        self,
        A,
        B,
        density_matrix_1: np.ndarray,
        density_matrix_2: np.ndarray,
        petz_map_result: np.ndarray,
    ):
        """tests the petz map using log version if there is no overlap"""
        precomp_dens_A = np_logm(density_matrix_1)
        precomp_dens_B = np_logm(density_matrix_2)

        petz_map = PetzMap(
            key_A=A,
            key_B=B,
            density_matrix_A=density_matrix_1,
            density_matrix_B=density_matrix_2,
            precomputed_sqrt_or_log_of_density_matrix_A=precomp_dens_A,
            precomputed_sqrt_or_log_of_density_matrix_B=precomp_dens_B,
            sqrt_method=False,
        )
        rho_AB = petz_map.get_combined_system()
        new_key = petz_map.get_new_key()
        assert np.allclose(rho_AB, petz_map_result)
        assert new_key.n == 1.5
        assert new_key.level == 1

    @pytest.mark.parametrize(
        "density_matrix_1, density_matrix_2, petz_map_result",
        [(np.eye(4) / 4, np.eye(4) / 4, np.eye(8) / 8)],
    )
    def test_petz_map_sqrt_overlap(
        self,
        density_matrix_1: np.ndarray,
        density_matrix_2: np.ndarray,
        petz_map_result: np.ndarray,
    ):
        """tests the petz map using sqrt version if there is overlap"""
        precomp_dens_A = np_sqrt(density_matrix_1)
        precomp_dens_B = np_sqrt(density_matrix_2)

        A = (1.5, 1)
        B = (2.5, 1)
        petz_map = PetzMap(
            key_A=A,
            key_B=B,
            density_matrix_A=density_matrix_1,
            density_matrix_B=density_matrix_2,
            precomputed_sqrt_or_log_of_density_matrix_A=precomp_dens_A,
            precomputed_sqrt_or_log_of_density_matrix_B=precomp_dens_B,
            sqrt_method=True,
        )
        rho_AB = petz_map.get_combined_system()
        new_key = petz_map.get_new_key()
        assert np.allclose(rho_AB, petz_map_result)
        assert new_key.n == 2
        assert new_key.level == 2

    @pytest.mark.parametrize(
        "density_matrix_1, density_matrix_2, petz_map_result",
        [(np.eye(4) / 4, np.eye(4) / 4, np.eye(8) / 8)],
    )
    def test_petz_map_log_overlap(
        self,
        density_matrix_1: np.ndarray,
        density_matrix_2: np.ndarray,
        petz_map_result: np.ndarray,
    ):
        """tests the petz map using log version if there is overlap"""
        precomp_dens_A = np_logm(density_matrix_1)
        precomp_dens_B = np_logm(density_matrix_2)

        A = (1.5, 1)
        B = (2.5, 1)
        petz_map = PetzMap(
            key_A=A,
            key_B=B,
            density_matrix_A=density_matrix_1,
            density_matrix_B=density_matrix_2,
            precomputed_sqrt_or_log_of_density_matrix_A=precomp_dens_A,
            precomputed_sqrt_or_log_of_density_matrix_B=precomp_dens_B,
            sqrt_method=True,
        )
        rho_AB = petz_map.get_combined_system()
        new_key = petz_map.get_new_key()
        assert np.allclose(rho_AB, petz_map_result)
        assert new_key.n == 2
        assert new_key.level == 2

    @pytest.mark.parametrize(
        "A, B, density_matrix_1, density_matrix_2",
        [
            (
                (1, 0),
                (2, 0),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.array([[0.5, 0.0], [0.0, 0.5]]),
            ),
            (
                (1, 0),
                (2, 0),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
                np.array([[0.9, 0.0], [0.0, 0.1]]),
            ),
            (
                (1, 0),
                (2, 0),
                np.array([[0.8, 0.0], [0.0, 0.2]]),
                np.array([[0.8, 0.0], [0.0, 0.2]]),
            ),
        ],
    )
    def test_ptrace(self, A, B, density_matrix_1, density_matrix_2):
        """tests if the correct subsystem density matrices are produced"""
        precomp_dens_A = np_sqrt(density_matrix_1)
        precomp_dens_B = np_sqrt(density_matrix_2)

        p_map = PetzMap(
            key_A=A,
            key_B=B,
            density_matrix_A=density_matrix_1,
            density_matrix_B=density_matrix_2,
            precomputed_sqrt_or_log_of_density_matrix_A=precomp_dens_A,
            precomputed_sqrt_or_log_of_density_matrix_B=precomp_dens_B,
            sqrt_method=True,
        )
        rho_AB = p_map.get_combined_system()

        assert np.allclose(density_matrix_1, ptrace(rho_AB, 1, end="right"))
        assert np.allclose(density_matrix_2, ptrace(rho_AB, 1, end="left"))

    @pytest.mark.parametrize(
        "system_size, density_matrix",
        [
            (
                4,
                np.array([[0.5, 0.0], [0.0, 0.5]]),
            ),
            (
                5,
                np.array([[0.9, 0.0], [0.0, 0.1]]),
            ),
            (
                6,
                np.array([[0.8, 0.0], [0.0, 0.2]]),
            ),
        ],
    )
    def test_ptrace_in_system_of_density_matrices(
        self, system_size: int, density_matrix: np.ndarray
    ):
        """
        tests if the correct subsystem density matrices
        are produced from a Petz-mapped density matrix
        """
        system = [density_matrix for _ in range(system_size)]
        old_system = []
        new_system = []
        for ell in range(system_size - 1):
            new_system = []
            for n in range(system_size - ell - 1):
                key_A = (n + ell / 2, ell)
                key_B = (n + ell / 2 + 1, ell)
                petz_map = PetzMap(
                    key_A=key_A,
                    key_B=key_B,
                    density_matrix_A=system[n],
                    density_matrix_B=system[n + 1],
                    sqrt_method=True,
                )
                new_system += [petz_map.get_combined_system()]
            old_system = deepcopy(system)
            system = new_system

        assert np.allclose(old_system[0], ptrace(new_system[0], 1, end="right"))
        assert np.allclose(old_system[1], ptrace(new_system[0], 1, end="left"))


@pytest.mark.parametrize(
    "rho_AB, mut_info",
    [
        (
            np.array(
                [
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.5],
                ]
            ),
            np.log(2),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.log(2),
        ),
        (
            np.array(
                [
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            0.0,
        ),
        (
            np.array(
                [
                    [0.25, 0.0, 0.0, 0.0],
                    [0.0, 0.25, 0.0, 0.0],
                    [0.0, 0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.0, 0.25],
                ]
            ),
            0.0,
        ),
        (np.eye(8) / 8, 0.0),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            0.0,
        ),
        (
            np.array(
                [
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.log(2),
        ),
    ],
)
def test_mutual_information(rho_AB, mut_info):
    rho_A = ptrace(rho_AB, 1, end="right")
    rho_B = ptrace(rho_AB, 1, end="left")
    if len(rho_B) > 2:
        rho_AnB = ptrace(rho_A, 1, end="left")
    else:
        rho_AnB = None

    computed_mut_info = mutual_information(rho_AB, rho_A, rho_B, rho_AnB)
    assert np.allclose(mut_info, computed_mut_info)
