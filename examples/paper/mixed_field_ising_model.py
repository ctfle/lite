import os

# This script generates some of the results shown in PRX QUANTUM 5, 020352 (2024)
# for the mixed field Ising model. Simulations with large min_l and max_l might require
# distributed computing systems to yield the long-time time-evolution.


# sets the number of threads used in the first place for numpy matrix operations
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import pickle
import local_information as li
import matplotlib.pyplot as plt

min_l = 4
max_l = 6
L = 101

J = 1.0
ht = 1.4
hl = 0.9045


J_list = [J for _ in range(L)]
ht_list = [ht for _ in range(L)]
hl_list = [hl for _ in range(L)]
hamiltonian_couplings = [["zz", J_list], ["x", ht_list], ["z", hl_list]]
setup_hamiltonian = li.operators.Hamiltonian(max_l, hamiltonian_couplings)

# locally thermal Hamiltonian
beta = 0.05
number_of_thermal_sites = 3
thermal_regime = (
    np.eye(2**number_of_thermal_sites, dtype=np.complex128)
    - beta
    * setup_hamiltonian.subsystem_hamiltonian[
        (L // 2, number_of_thermal_sites - 1)
    ].toarray()
) / 2**number_of_thermal_sites
# infinite temperature sites on the boundary
infinite_temperature_site = np.array([[0.5, 0.0], [0.0, 0.5]])
initial_state = li.State.build([[thermal_regime], [infinite_temperature_site]], 1)


# Additional observable: x and y magnetization
x_ = [1.0 for j in range(L)]
x_list = [["x", x_]]
x_mag = li.operators.Operator(x_list)


def x_magnetization(rho):
    return x_mag.expectation_value(rho)


z_ = [1.0 for j in range(L)]
z_list = [["z", z_]]
z_mag = li.operators.Operator(z_list)


def z_magnetization(rho):
    return z_mag.expectation_value(rho)


checkpoint_folder = f"mixed_field_ising_ht={ht}_hl={hl}_J={J}_L={L}_diffusion"
data_config = li.DataConfig(
    observables=[z_magnetization, x_magnetization],
    diffusion_const=True,
    info_lattice=True,
)
data = li.DataContainer(config=data_config)
config = li.config.TimeEvolutionConfig(
    save_checkpoint=True,
    checkpoint_folder=checkpoint_folder,
    min_l=min_l,
    max_l=max_l,
    shift=10,
)

system = li.ClosedSystem(initial_state, setup_hamiltonian, config=config, data=data)

steps = 10
for i in range(steps):
    # final_time should be a bool value
    system.evolve(max_evolution_time=2.0, final_time=True)
    print(f"finished cycle {i} of {steps}")
    system.solver.step_size = 0.25


def load_from_file(file_path: str):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


# Load the data
data_filepath = os.path.join(os.getcwd(), checkpoint_folder)
diffusion_const = load_from_file(os.path.join(data_filepath, "diffusion_const.pkl"))
diffusion_length = load_from_file(os.path.join(data_filepath, "diffusion_length.pkl"))
times = load_from_file(os.path.join(data_filepath, "times.pkl"))

# Plot the diffusion constant over time
plt.plot(times, diffusion_const, color="black", label="diffusion const")
plt.title(f"Diffusion constant")
plt.xlabel("t")
plt.ylabel("D(t)")
plt.xscale("log")
plt.show()


# Plot the diffusion length over time
plt.plot(times, diffusion_length, color="black", label="diffusion const")
plt.title(f"Diffusion length")
plt.xlabel("t")
plt.ylabel("D(t)")
plt.xscale("log")
plt.show()
