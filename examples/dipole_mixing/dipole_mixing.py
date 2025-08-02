### This script reproduces the dissipative dynamics shown in PRX QUANTUM 5, 020352 (2024) ###

import os

# Sets the number of threads (mostly used in numpy matrix operations)
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import pickle
import local_information as li
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm


# Set some parameters:
# min_l and max_l describe the min and max correlation length scales at which
# the algorithm operates: minimization of information is triggered as soon as
# the enough max_l-local information has build up.
min_l = 3
max_l = 6
L = 301
dissipation_strength = 0.1
J = 1.0
P = 11


couplings = [J for j in range(L)]
hamiltonian_couplings = [["xx", couplings], ["yy", couplings]]

# Lindblad part for the evolution
jumps = [dissipation_strength for _ in range(L)]
jump_couplings = [["z", jumps]]

setup_lindbladian = li.operators.Lindbladian(
    max_l, hamiltonian_couplings, jump_couplings
)

# Initial state - We use a product state of single site density matrices
prob_up = 0.8
polarized_up_site = np.array([[prob_up, 0.0], [0.0, 1 - prob_up]])
polarized_down_site = np.array([[1 - prob_up, 0.0], [0.0, prob_up]])
infinite_temperature_site = np.array([[0.5, 0.0], [0.0, 0.5]])
boundary = [infinite_temperature_site for _ in range(2)]

bulk = []
for j in range(3 * P):
    if j <= P:
        bulk.append(polarized_up_site)
    elif P < j <= 2 * P:
        bulk.append(infinite_temperature_site)
    else:
        bulk.append(polarized_down_site)


# Build the initial state from a polarized part and an infinite temperature boundary
# reaching out to +/- infinity
initial_state = li.State.build_asymptotic(bulk, boundary, 1)


# Additional observables: z-magnetisation and a diffusion constant representing magnetisation diffusion
z = [1.0 for j in range(L)]
z_list = [["z", [1.0 for j in range(L)]]]
z_mag = li.operators.Operator(z_list)


def z_magnetization(rho):
    return z_mag.expectation_value(rho)


def z_mag_diff_const(rho):
    z_operator_current = setup_lindbladian.operator_current(z_mag)
    return li.operators.observables.onsite_operator_diff_const(
        rho,
        z_mag.operator,
        z_operator_current,
        initial_state.anchor,
    )


# directory to store the data
checkpoint_folder = f"dipole_mixing_dissipation={dissipation_strength}_J={J}_L={L}"

# DataConfig and DataContainer is needed to communicate which observables to measure and checkpoint
data_config = li.DataConfig(
    observables=[z_magnetization, z_mag_diff_const],
    diffusion_const=True,
    info_lattice=True,
)
data = li.DataContainer(config=data_config)

# A TimeEvolutionConfig where all parameters relevant for time evolution are set
config = li.config.TimeEvolutionConfig(
    save_checkpoint=True,
    checkpoint_folder=checkpoint_folder,
    min_l=min_l,
    max_l=max_l,
    shift=10,
)

# Define the system from an initial state, a Hamiltonian/ Lindbladian and the config's
system = li.OpenSystem(initial_state, setup_lindbladian, config=config, data=data)

# the main time-evolution loop. Observables and state are checkpointed after each iteration
steps = 20
for i in range(steps):
    system.evolve(max_evolution_time=2.0, final_time=True)
    print(f"finished cycle {i} of {steps}")
    system.solver.step_size = 0.25


#######################################################################################
########################## loading and plotting the data ##############################
#######################################################################################


def get_time_and_space_resolved_z_magentization(loaded_z_mag: list[dict]):
    nr_of_measured_times = len(loaded_z_mag)
    # initialise an array with zeros
    magnetization = np.zeros((nr_of_measured_times, L))
    for time_step, z_mag_at_time in enumerate(loaded_z_mag):
        for (n, ell), mag in z_mag_at_time[0].items():
            magnetization[time_step, int(n)] = mag
    return magnetization


def load_from_file(file_path: str):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


# Load the data
data_filepath = os.path.join(os.getcwd(), checkpoint_folder)
z_magnetization = load_from_file(os.path.join(data_filepath, "z_magnetization.pkl"))
z_mag_diff_const = load_from_file(os.path.join(data_filepath, "z_mag_diff_const.pkl"))
times = load_from_file(os.path.join(data_filepath, "times.pkl"))


# Plot the z-magnetisation diffusion constant over time
plt.plot(times, z_mag_diff_const, color="black", label="z mag diffusion const")
plt.title(f"Diffusion constant for z mag")
plt.xlabel("t")
plt.ylabel("D(t)")
plt.xscale("log")
plt.show()


# Plot the space resolved magnetization as a function of time
time_and_space_resolved_z_magnetization = get_time_and_space_resolved_z_magentization(
    z_magnetization
)
number_of_slices = len(time_and_space_resolved_z_magnetization)
for time_slice in time_and_space_resolved_z_magnetization:
    plt.plot(time_slice)
plt.show()


# Animate the space resolved magnetization over time
fig, ax = plt.subplots(figsize=(8, 6))


def animate(i: int):
    ax.clear()
    cmap = cm.get_cmap("viridis")
    for j in range(i + 1):
        color = cmap(j / number_of_slices)
        ax.plot(time_and_space_resolved_z_magnetization[j], color=color)
    ax.set_ylim([-0.75, 0.75])
    ax.set_title(f"Time {np.round(times[i + 1], 3)} $J^{-1}$")


# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=number_of_slices, interval=100)
plt.show()
