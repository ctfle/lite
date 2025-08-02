from importlib import resources

import numpy as np


def load_RK12_coffs():
    """load the runge kutta coefficients for RK 12 (10)"""
    data_path = resources.path("local_information.core.runge_kutta_solvers", "RK12.txt")
    with data_path as path:
        with open(path) as RK:
            content = RK.readlines()

    b = []
    c = []
    a = np.zeros((25, 25))
    int_list = ["{0:d}".format(j) for j in range(10)]
    for row, con in enumerate(content):
        try:
            if not (con[3] in int_list):
                continue
            else:
                if row < 31:
                    c += [float(con[7 : 7 + 20])]
                elif 32 < row < 59:
                    b += [float(con[7 : 7 + 20])]
                elif 61 < row < 365:
                    a[int(con[2:4]), int(con[7:9])] = float(con[12 : 12 + 20])
        except:
            continue

    return c, a, b


# load the runge kutta coefficients for RK 12 (10)
def load_RK10_coffs():
    data_path = resources.path("local_information.core.runge_kutta_solvers", "RK10.txt")
    with data_path as path:
        with open(path) as RK:
            content = RK.readlines()
    b = []
    c = []
    a = np.zeros((17, 17))
    int_list = ["{0:d}".format(j) for j in range(10)]
    for row, con in enumerate(content):
        try:
            if not (con[1] in int_list):
                continue
            else:
                if row < 24:
                    c += [float(con[5 : 5 + 20])]
                elif 26 < row < 45:
                    b += [float(con[5 : 5 + 20])]
                elif 47 < row < 186:
                    a[int(con[:2]), int(con[5:8])] = float(con[10 : 10 + 20])
        except:
            continue
    return c, a, b


def RK45_parameters():
    # Runge Kutta  5 (4) coefficients
    c = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
    a = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 4, 0, 0, 0, 0, 0],
            [3 / 32, 9 / 32, 0, 0, 0, 0],
            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
            [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
        ]
    )
    b5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    b4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])

    return c, a, b5, b4


def RK23_parameters():
    # Runge Kutta  3 (2) coefficients
    c = np.array([0, 1 / 2, 3 / 4, 1])
    a = np.array(
        [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 3 / 4, 0, 0], [2 / 9, 1 / 3, 4 / 9, 0]]
    )
    b3 = np.array([2 / 9, 1 / 3, 4 / 9, 0])
    b2 = np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8])
    return c, a, b3, b2


def RK1012_parameters():
    # Runge Kutta  12 (10) coefficients
    c, a, b = load_RK12_coffs()
    return c, a, b


def RK810_parameters():
    # Runge Kutta  12 (10) coefficients
    c, a, b = load_RK10_coffs()
    return c, a, b
