import numpy as np
import pytest
from mock import MagicMock

from local_information.operators.operator import (
    Operator,
    check_hamiltonian,
    get_couplings,
    construct_operator_dict,
    check_lindbladian,
    setup_onsite_L_operators,
    construct_lindbladian_id,
    construct_lindbladian_dict,
)


class TestOperator:
    @pytest.fixture(scope="function")
    def mock_rho_dict(self, request):
        """mock rho_dict"""
        density_matrix, n_min, n_max = request.param
        mock = MagicMock()
        mock.__getitem__.return_value = density_matrix
        mock.boundaries.return_value = (n_min, n_max)
        return mock

    @pytest.fixture
    def test_two_body_operator(self):
        L = 10
        J = 0.25
        hL = 0.125
        hT = -0.2625
        J_list = [J for _ in range(L)]
        hT_list = [hT for _ in range(L)]
        hL_list = [hL for _ in range(L)]
        couplings = [["zz", J_list], ["z", hL_list], ["x", hT_list]]
        operator = Operator(couplings, name="test_two_body_operator")
        return operator

    @pytest.fixture
    def test_onsite_operator(self):
        L = 10
        hT = -0.2625
        hT_list = [hT for _ in range(L)]
        couplings = [["x", hT_list]]
        operator = Operator(couplings, name="test_onsite_operator")
        return operator

    @pytest.fixture
    def test_onsite_z_operator(self):
        L = 10
        hT = 1.0
        hT_list = [hT for _ in range(L)]
        couplings = [["z", hT_list]]
        operator = Operator(couplings, name="test_onsite_operator")
        return operator

    @pytest.fixture
    def test_onsite_zz_operator(self):
        L = 10
        hT = 1.0
        hT_list = [hT for _ in range(L)]
        couplings = [["zz", hT_list]]
        operator = Operator(couplings, name="test_onsite_operator")
        return operator

    def test_operator_range(self, test_onsite_operator, test_two_body_operator):
        assert test_onsite_operator.operator.dim_at_level(1) == 0
        assert test_onsite_operator.operator.dim_at_level(0) == 10

        assert list(test_two_body_operator.operator.keys_at_level(0)) == []
        assert list(test_two_body_operator.operator.keys_at_level(1)) != []
        assert test_two_body_operator.operator.dim_at_level(1) == 9
        pass

    def test_eq(self, test_onsite_operator, test_two_body_operator):
        assert test_onsite_operator != test_two_body_operator
        assert test_onsite_operator == test_onsite_operator
        assert test_two_body_operator == test_two_body_operator
        pass

    @pytest.mark.parametrize(
        "mock_rho_dict", [(np.eye(2) / 2, 0.0, 9.0)], indirect=["mock_rho_dict"]
    )
    def test_inf_temp_expectation_value_single_particle_operator(
        self, mock_rho_dict, test_onsite_operator
    ):
        # tests expectation value
        expt_val_dict, expt_val = test_onsite_operator.expectation_value(mock_rho_dict)
        assert expt_val == 0.0
        for key, val in expt_val_dict.items():
            assert val == 0.0
        pass

    @pytest.mark.parametrize(
        "mock_rho_dict", [(np.eye(4) / 4, 0.5, 8.5)], indirect=["mock_rho_dict"]
    )
    def test_inf_temp_expectation_value_two_particle_operator(
        self, mock_rho_dict, test_two_body_operator
    ):
        # tests expectation value
        expt_val_dict, expt_val = test_two_body_operator.expectation_value(
            mock_rho_dict
        )
        assert expt_val == 0.0
        for key, val in expt_val_dict.items():
            assert val == 0.0
        pass

    @pytest.mark.parametrize(
        "mock_rho_dict, value",
        [
            ((np.array([[1, 0], [0, 0]]), 0.0, 9.0), 1.0),
            ((np.array([[0.9, 0], [0, 0.1]]), 0.0, 9.0), 0.9 - 0.1),
            ((np.array([[0.8, 0], [0, 0.2]]), 0.0, 9.0), 0.8 - 0.2),
            ((np.array([[0.7, 0], [0, 0.3]]), 0.0, 9.0), 0.7 - 0.3),
        ],
        indirect=["mock_rho_dict"],
    )
    def test_expectation_value_single_particle_operator(
        self, mock_rho_dict, value, test_onsite_z_operator
    ):
        # tests expectation value
        expt_val_dict, expt_val = test_onsite_z_operator.expectation_value(
            mock_rho_dict
        )
        assert np.allclose(expt_val, 10 * value)
        for key, val in expt_val_dict.items():
            assert np.allclose(val, value)
        pass

    @pytest.mark.parametrize(
        "mock_rho_dict, value",
        [
            (
                (
                    np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                    0.5,
                    8.5,
                ),
                1.0,
            ),
            (
                (
                    np.array(
                        [[0.9, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                    ),
                    0.5,
                    8.5,
                ),
                0.9 - 0.1,
            ),
            (
                (
                    np.array(
                        [[0.8, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                    ),
                    0.5,
                    8.5,
                ),
                0.8 - 0.2,
            ),
            (
                (
                    np.array(
                        [[0.4, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.4]]
                    ),
                    0.5,
                    8.5,
                ),
                0.8 - 0.2,
            ),
            (
                (
                    np.array(
                        [[0.3, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0.2, 0], [0, 0, 0, 0.3]]
                    ),
                    0.5,
                    8.5,
                ),
                0.6 - 0.4,
            ),
        ],
        indirect=["mock_rho_dict"],
    )
    def test_expectation_value_two_body_operator(
        self, mock_rho_dict, value, test_onsite_zz_operator
    ):
        # tests expectation value
        expt_val_dict, expt_val = test_onsite_zz_operator.expectation_value(
            mock_rho_dict
        )
        assert np.allclose(expt_val, 9 * value)
        for _, val in expt_val_dict.items():
            assert np.allclose(val, value)

    @pytest.mark.parametrize(
        "couplings, allowed_strings, coupling_range",
        [
            (
                [["zz", [1 for _ in range(10)]], ["z", [1 for _ in range(10)]]],
                ["x", "y", "z", "1"],
                1,
            ),
            (
                [["zyz", [1 for _ in range(10)]], ["z", [1 for _ in range(10)]]],
                ["x", "y", "z", "1"],
                2,
            ),
            (
                [["zyxxy", [1 for _ in range(10)]], ["z", [1 for _ in range(10)]]],
                ["x", "y", "z", "1"],
                4,
            ),
        ],
    )
    def test_check_hamiltonian(self, couplings, allowed_strings, coupling_range):
        range_, disorder = check_hamiltonian(couplings, allowed_strings)
        assert range_ == coupling_range
        for element in disorder:
            assert not any(element)
        pass

    @pytest.mark.parametrize(
        "couplings, allowed_strings",
        [
            (
                [["za", [1 for _ in range(10)]], ["z", [1 for _ in range(10)]]],
                ["x", "y", "z", "1"],
            ),
            (
                [["abc", [1 for _ in range(10)]], ["z", [1 for _ in range(10)]]],
                ["x", "y", "z", "1"],
            ),
        ],
    )
    def test_invalid_hamiltonian(self, couplings, allowed_strings):
        with pytest.raises(ValueError):
            check_hamiltonian(couplings, allowed_strings)

    @pytest.mark.parametrize(
        "couplings, boundaries, operator_range, value",
        [
            ([["z", [1.2 for _ in range(2)]]], (0, 2), 0, 1.2),
            ([["z", [1.2 for _ in range(4)]]], (0, 4), 0, 1.2),
            (
                [["zz", [1.3 for _ in range(4)]], ["xx", [1.3 for _ in range(4)]]],
                (0, 4),
                1,
                1.3,
            ),
        ],
    )
    def test_get_couplings(self, couplings, boundaries, operator_range, value):
        n_min = boundaries[0]
        n_max = boundaries[1]
        quspin_couplings = get_couplings(
            operator_couplings=couplings, n_min=n_min, n_max=n_max
        )
        for coupling_element in quspin_couplings:
            assert len(coupling_element[1]) == n_max - operator_range
            for part in coupling_element[1]:
                assert part[0] == value
                assert len(part) == operator_range + 2
                if operator_range > 0:
                    for i, index in enumerate(part[1:-1]):
                        assert index == part[i + 1]
        pass

    @pytest.mark.parametrize(
        "couplings, max_l, range_, system_size",
        [
            ([["zyx", [1.2 for _ in range(10)]]], 5, 3, 10),
            (
                [["zy", [1.2 for _ in range(10)]], ["zx", [1.2 for _ in range(10)]]],
                4,
                0,
                10,
            ),
            (
                [["zy", [1.2 for _ in range(10)]], ["zxxx", [1.2 for _ in range(10)]]],
                4,
                0,
                10,
            ),
        ],
    )
    def test_construct_operator_dict(self, couplings, max_l, range_, system_size):
        op_dict = construct_operator_dict(couplings, max_l, range_, system_size)

        assert op_dict.keys_at_level(max_l + range_) is not None
        for key, val in op_dict.items_at_level(0):
            assert np.allclose(val.toarray(), np.zeros((2, 2)))

        for ell in range(max_l + range_):
            assert op_dict.dim_at_level(ell) == system_size - ell
        pass

    @pytest.mark.parametrize(
        "couplings, range_, system_size",
        [
            ([["zyx", [1.2 for _ in range(10)]]], 2, 10),
            (
                [["zy", [1.2 for _ in range(10)]], ["zx", [1.2 for _ in range(10)]]],
                1,
                10,
            ),
            (
                [["zy", [1.2 for _ in range(10)]], ["zxxx", [1.2 for _ in range(10)]]],
                3,
                10,
            ),
        ],
    )
    def test_local_operator_dict(self, couplings, range_, system_size):
        op = Operator(couplings)
        assert op.operator.keys_at_level(range_) is not None
        for ell in range(range_):
            if ell != range_:
                assert op.operator.dim_at_level(ell) == 0
            else:
                assert op.operator.dim_at_level(ell) == system_size - ell

        assert op.range_ == range_
        assert op.L == system_size

        pass

    @pytest.mark.parametrize(
        "couplings1, couplings2",
        [
            ([["z", [1.2 for _ in range(10)]]], [["y", [1.2 for _ in range(10)]]]),
            ([["zz", [1.2 for _ in range(10)]]], [["yy", [1.2 for _ in range(10)]]]),
            ([["zzz", [1.2 for _ in range(10)]]], [["yxy", [1.2 for _ in range(10)]]]),
        ],
    )
    def test_operator_addition(self, couplings1, couplings2):
        op1 = Operator(couplings1)
        op2 = Operator(couplings2)
        sumed_op = Operator(couplings1 + couplings2)
        assert op1 + op2 == sumed_op

    @pytest.mark.parametrize(
        "coupling1, coupling2",
        [
            ([["z", [1.2 for _ in range(10)]]], [["z", [1.2 for _ in range(10)]]]),
            ([["zz", [1.2 for _ in range(10)]]], [["zz", [1.2 for _ in range(10)]]]),
            ([["yyy", [1.2 for _ in range(10)]]], [["yyy", [1.2 for _ in range(10)]]]),
        ],
    )
    def test_operator_subtraction(self, coupling1, coupling2):
        op1 = Operator(coupling1)
        op2 = Operator(coupling2)
        sub_operator = op1 - op2
        for key, val in sub_operator.operator.items():
            assert np.sum(val.toarray()) == 0.0
        pass

    @pytest.mark.parametrize(
        "couplings, scalar",
        [
            ([["z", [1.0 for _ in range(10)]]], 3.12345),
            ([["zz", [1.2 for _ in range(10)]]], 6.12 + 1.23 * 1j),
            ([["yyy", [1.2 for _ in range(10)]]], 4),
        ],
    )
    def test_scalar_multiplication(self, couplings, scalar):
        op1 = Operator(couplings)
        op1 * scalar
        mult_coupling = []
        for _, coupling_element in enumerate(couplings):
            multiplied = [scalar * element for element in coupling_element[1]]

            mult_coupling += [[coupling_element[0], multiplied]]

        mult_operator = Operator(mult_coupling)
        assert op1 == mult_operator

    @pytest.mark.parametrize(
        "jump_couplings, allowed_strings, type_set",
        [
            (
                [["z", [1 for _ in range(10)]], ["x", [1 for _ in range(10)]]],
                ["x", "y", "z", "+", "-", "1"],
                {"z", "x"},
            )
        ],
    )
    def test_check_lindbladian(self, jump_couplings, allowed_strings, type_set):
        range_, types, disorder = check_lindbladian(jump_couplings, allowed_strings)
        assert type_set == set(types)
        for element in disorder:
            assert not any(element)
        pass

    @pytest.mark.parametrize(
        "max_l, range_, type_list",
        [(1, 1, ["z"]), (3, 1, ["z"]), (3, 1, ["x", "y", "z"])],
    )
    def test_setup_onsite_L_operators(self, max_l, range_, type_list):
        """setup_onsite_L_operators creates all possible single particle operators
        for later use.
        """
        L_operators = setup_onsite_L_operators(max_l, range_, type_list)
        for key, val in L_operators.items():
            assert key[2] in ["x", "y", "z", "+", "-", "1"]
            assert key[0] >= key[1]
        # ensure that all keys exist
        for extend in range(max_l + range_):
            for m in range(extend):
                for typ in type_list:
                    assert (extend, m, typ) in L_operators.keys()
        pass

    @pytest.mark.parametrize(
        "n_max, jump_couplings, n_min, jump_signature",
        [
            (10, [["z", [1 for _ in range(10)]]], 0, None),
            (
                10,
                [["y", [0.1 if j == 10 // 2 else 0.0 for j in range(10)]]],
                0,
                (5, 0.1, ["y"]),
            ),
            (
                10,
                [["y", [0.1 if j == 10 // 2 else None for j in range(10)]]],
                0,
                (5, 0.1, ["y"]),
            ),
            (
                10,
                [
                    ["y", [0.1 if j == 10 // 2 else None for j in range(10)]],
                    ["x", [0.1 if j == 10 // 2 else None for j in range(10)]],
                    ["z", [0.1 if j == 10 // 2 else None for j in range(10)]],
                ],
                0,
                (5, 0.1, ["x", "y", "z"]),
            ),
            (
                10,
                [
                    ["+", [0.1 if j == 10 // 2 else None for j in range(10)]],
                    ["-", [0.1 if j == 10 // 2 else None for j in range(10)]],
                    ["z", [0.1 if j == 10 // 2 else None for j in range(10)]],
                ],
                0,
                (5, 0.1, ["+", "-", "z"]),
            ),
        ],
    )
    def test_construct_lindbladian_id_1(
        self, n_max, jump_couplings, n_min, jump_signature
    ):
        """construct_lindbladdian_id holds information
        where to apply onsite Lindblad operators"""
        id_list = construct_lindbladian_id(n_max, jump_couplings, n_min)
        for i, id_ in enumerate(id_list):
            if jump_signature is not None:
                (jump_index, jump_value, jump_type) = jump_signature
                if i != jump_index:
                    assert id_ is None
                else:
                    for id_element in id_:
                        assert id_element[0] in jump_type
                        assert id_element[1] == jump_value
        pass

    @pytest.mark.parametrize(
        "n_max, jump_couplings, n_min, jump_signature",
        [
            (
                10,
                [
                    ["y", [0.1 if j == 10 // 2 else None for j in range(10)]],
                    ["x", [0.1 if j == 10 // 2 else None for j in range(10)]],
                    ["z", [1.2 for _ in range(10)]],
                ],
                0,
                ([0.1, 1.2], ["x", "y", "z"]),
            )
        ],
    )
    def test_construct_lindbladian_id_2(
        self, n_max, jump_couplings, n_min, jump_signature
    ):
        """construct_lindbladdian_id holds information
        where to apply onsite Lindblad operators"""
        id_list = construct_lindbladian_id(n_max, jump_couplings, n_min)
        for i, id_ in enumerate(id_list):
            if i != 5:
                for id_element in id_:
                    assert id_element[0] == "z"
                    assert id_element[1] == 1.2
            else:
                for id_element in id_:
                    (jump_value, jump_type) = jump_signature
                    assert id_element[0] in jump_type
                    assert id_element[1] in jump_value
        pass

    @pytest.mark.parametrize(
        "jump_couplings, max_l, range_, system_size, expected",
        [
            ([["z", [1.23445 for _ in range(3)]]], 1, 1, 3, ("z", 1.23445)),
            ([["y", [1.25 for _ in range(30)]]], 1, 1, 30, ("y", 1.25)),
            ([["x", [3.245 for _ in range(10)]]], 1, 1, 10, ("x", 3.245)),
            ([["+", [3.245 for _ in range(10)]]], 1, 1, 10, ("+", 3.245)),
            ([["-", [3.245 for _ in range(10)]]], 1, 1, 10, ("-", 3.245)),
        ],
    )
    def test_construct_lindbladian_dict(
        self, jump_couplings, max_l, range_, system_size, expected
    ):
        """! Constructs a dictionary that holds the information
        which on-site Lindblad operator to apply where"""

        lindblad_dict = construct_lindbladian_dict(
            jump_couplings, max_l, range_, system_size
        )
        for key, val in lindblad_dict.items():
            assert len(val) == key[1] + 1
            for block in val:
                for element in block:
                    assert element == expected
        pass

    @pytest.mark.parametrize(
        "jump_couplings, max_l, range_, system_size, jump_site",
        [
            ([["+", [0.1 if j == 2 else None for j in range(4)]]], 1, 1, 4, 2),
            ([["-", [0.1 if j == 2 else None for j in range(4)]]], 1, 1, 4, 2),
            ([["z", [0.1 if j == 2 else None for j in range(4)]]], 1, 1, 4, 2),
        ],
    )
    def test_lindbladian_dict_1(
        self, jump_couplings, max_l, range_, system_size, jump_site
    ):
        """! Constructs a dictionary that holds the information
        which on-site Lindblad operator to apply where"""
        lindblad_dict = construct_lindbladian_dict(
            jump_couplings, max_l, range_, system_size
        )
        for key, val in lindblad_dict.items():
            n = key[0]
            ell = key[1]
            if n + ell / 2 >= jump_site >= n - ell / 2:
                for i, index in enumerate(
                    range(int(n - ell / 2), int(n + ell / 2) + 1)
                ):
                    if index == jump_site:
                        assert val[i] is not None
            else:
                assert val[0] is None
            pass

    @pytest.mark.parametrize(
        "jump_couplings, max_l, range_, system_size, expected",
        [
            (
                [
                    ["+", [0.1 if j == 2 else None for j in range(4)]],
                    ["-", [0.1 if j == 2 else None for j in range(4)]],
                    ["z", [0.1 if j == 2 else None for j in range(4)]],
                ],
                1,
                1,
                4,
                (["+", "-", "z"], 0.1, 2),
            ),
        ],
    )
    def test_lindbladian_dict_2(
        self, jump_couplings, max_l, range_, system_size, expected
    ):
        """! Constructs a dictionary that holds the information
        which on-site Lindblad operator to apply where"""
        jump_site = expected[2]
        lindblad_dict = construct_lindbladian_dict(
            jump_couplings, max_l, range_, system_size
        )
        for key, val in lindblad_dict.items():
            n = key[0]
            ell = key[1]
            if n + ell / 2 >= jump_site >= n - ell / 2:
                for i, index in enumerate(
                    range(int(n - ell / 2), int(n + ell / 2) + 1)
                ):
                    if index == jump_site:
                        assert val[i] is not None
                        for element in val[i]:
                            assert element[0] in expected[0]
                            assert element[1] == expected[1]
            else:
                assert val[0] is None
            pass

    @pytest.mark.parametrize(
        "jump_couplings, max_l, range_, system_size, expected",
        [
            (
                [
                    ["+", [0.1 for j in range(4)]],
                    ["-", [0.1 if j == 2 else None for j in range(4)]],
                    ["z", [0.1 if j == 3 else None for j in range(4)]],
                ],
                1,
                1,
                4,
                (["+", "-", "z"], 0.1, 2),
            ),
        ],
    )
    def test_lindbladian_dict_3(
        self, jump_couplings, max_l, range_, system_size, expected
    ):
        """! Constructs a dictionary that holds the information
        which on-site Lindblad operator to apply where"""
        jump_value = expected[1]
        jump_site = expected[2]
        lindblad_dict = construct_lindbladian_dict(
            jump_couplings, max_l, range_, system_size
        )
        for key, val in lindblad_dict.items():
            n = key[0]
            ell = key[1]
            # each element in the val must contain the '+' term
            for element in val:
                assert ("+", jump_value) in element

            # the other terms only appear at the site where we added them
            if n + ell / 2 >= jump_site >= n - ell / 2:
                for i, index in enumerate(
                    range(int(n - ell / 2), int(n + ell / 2) + 1)
                ):
                    if index == jump_site:
                        assert val[i] is not None
                        for element in val[i]:
                            assert element[0] in expected[0]
                            assert element[1] == expected[1]
        pass
