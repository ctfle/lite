import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

L = 301
dissipation_strength = 0.1
J = 1.0
checkpoint_folder = f"dipole_mixing_dissipation={dissipation_strength}_J={J}_L={L}"


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
