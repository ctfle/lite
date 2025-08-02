import os, sys

os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from pathlib import Path
import time
import pickle
import src as li
from local_information.mpi.mpi_funcs import get_mpi_variables

COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()
import matplotlib.pyplot as plt

min_l = 4
max_l = 6
L = 1001
dissipation_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
dissipation_strength = dissipation_list[int(sys.argv[1])]

J_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
P = 11
J = J_list[int(sys.argv[2])]
hz = 0.2

J_list = [J for j in range(L)]
z_list = [hz for j in range(L)]
hamiltonian_couplings = [["xx", J_list], ["yy", J_list], ["z", z_list]]


# lindblad part for the evolution
plus_list = [dissipation_strength for j in range(L)]
minus_list = [dissipation_strength for j in range(L)]
z_list = [dissipation_strength for j in range(L)]
jump_couplings = [["z", z_list]]  # [['+',plus_list],['-',minus_list], ['z',z_list]]


# lindbaldians
lindbladian_Hdd = li.src.operators.Lindbladian(
    max_l, hamiltonian_couplings, jump_couplings
)


# intial state - product state of single site density matrices
prob1x = 0.7
prob2x = 0.3
prob_up = prob1x - prob2x
prob_down = -0.2  # prob2x - prob1x


site_1 = np.array([[prob_up, 0.0], [0.0, 1 - prob_up]])
site_0 = np.array([[0.5, 0.0], [0.0, 0.5]])


# beta = 1.0
# site_1 = (np.eye(2**3, dtype=np.complex128) - beta * lindbladian_Hdd.hamiltonian_dict[(L//2,2)].toarray())/8

# homogeneous state
n_plus = P
state = [site_1 for _ in range(P)]

dens_matrix_list = [
    state,
    [site_0],
]  # the first "structure_part", i.e., NOT translationally invariant. The second is the TI part
initial_state = li.src.State.build(dens_matrix_list, 1)

# x-magnetisation
z_ = [1.0 for j in range(L)]
z_list = [["z", z_]]
z_mag = li.src.operators.Operator(z_list)


# observables
def z_magnetization(rho):
    # return expect_value(rho, magnetization_dict, 0)
    return z_mag.expectation_value(rho)


z_operator_current = lindbladian_Hdd.operator_current(z_mag)


def z_mag_diff_cosnt(rho):
    return li.src.operators.observables.onsite_operator_diff_const(
        rho,
        z_mag.operator,
        lindbladian_Hdd.range_,
        z_operator_current,
        initial_state.n0,
    )


def save(data, filepath):
    with open(filepath, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


## time evolution loop
checkpoint_folder = (
    f"./xx_yy_z={hz}_dissipation={dissipation_strength}_J={J}_L={L}_z_diffusion"
)


data_config = li.src.config.DataConfig(observables=[z_magnetization, z_mag_diff_cosnt])
data = li.config.DataContainer(config=data_config)


config = li.src.config.TimeEvolutionConfig(
    save_checkpoint=True, checkpoint_folder=checkpoint_folder, min_l=min_l, max_l=max_l
)


system = li.src.OpenSystem(initial_state, lindbladian_Hdd, config=config, data=data)
# print(system)

rk_config = li.src.config.RungeKuttaConfig(step_size=0.1)

steps = 50
for i in range(steps):
    system.evolve(max_evolution_time=5, final_time=True)
    print(f"finished cycle {i} of {steps}")
    system.solver.step_size = 0.25

"""
magnetisation = system.data.custom_observables_dict['z_magnetization']
tot_mag_2 = np.zeros(len(magnetisation))
mag_vs_time_2 = np.zeros((len(magnetisation),L))
for t, mag in enumerate(magnetisation):
	#for n in range(L):
	#	mag_vs_time_2[t,n] = np.real(magnetization_2[t][(n,0)])
	for key in mag[0]:

		n = int(key[0])
		mag_vs_time_2[t,n] = np.real(mag[0][key])
	tot_mag_2[t]= np.sum(mag_vs_time_2[t])

for t, mag in enumerate(mag_vs_time_2):
	plt.plot(mag)
plt.show()

plt.plot(tot_mag_2)
plt.show()

diff_const = system.data.default_observables_dict['diffusion_const']
plt.plot(diff_const)
plt.show()
"""
