import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def get_time_and_space_resolved_x_magentization(loaded_x_mag: list[dict]):
    nr_of_measured_times = len(loaded_x_mag)
    # initialise an array with zeros
    magnetization = np.zeros((nr_of_measured_times, L))
    for time_step, x_mag_at_time in enumerate(loaded_x_mag):
        for (n, ell), mag in x_mag_at_time[0].items():
            magnetization[time_step, int(n)] = mag
    return magnetization


def load_from_file(file_path: str):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data

checkpoint_folder='./data/2_NV_diffusion=0.0_J=-0.1_L=501_rtrunc=0.2_mean_dis=5.0_delta_phi=0.1_num_pol=21'
L = 501

# Load the data
data_filepath = os.path.join(os.getcwd(), checkpoint_folder)
x_magnetization = load_from_file(os.path.join(data_filepath, "x_magnetization.pkl"))
times = load_from_file(os.path.join(data_filepath, "times.pkl"))


# Plot the space resolved magnetization as a function of time
time_and_space_resolved_x_magnetization = get_time_and_space_resolved_x_magentization(
    x_magnetization
)
number_of_slices = len(time_and_space_resolved_x_magnetization)
for time_slice in time_and_space_resolved_x_magnetization:
    plt.plot(time_slice)
plt.show()


total_x_mag = []
for t in range(len(time_and_space_resolved_x_magnetization)):
    total_x_mag.append(np.real(np.sum(time_and_space_resolved_x_magnetization[t])))


fig, ax = plt.subplots()
inset_ax = inset_axes(ax, width="30%", height="30%",
                      bbox_to_anchor=(0.65, 0.58, 0.9, 1.0),
                      bbox_transform=ax.transAxes,
                      loc='lower left')

frame_pointer = [0]
def animate_one_at_a_time(i: int):
    ax.clear()
    inset_ax.clear()
   
    y = time_and_space_resolved_x_magnetization[frame_pointer[0]]
    x = np.arange(len(y))
    line, = ax.plot(x, y, color="black")
    # Area above 0 -> light red
    ax.fill_between(x, y, 0, where=(y > 0), facecolor='red', alpha=0.3, interpolate=True)
    # Area below 0 -> light blue
    ax.fill_between(x, y, 0, where=(y < 0), facecolor='blue', alpha=0.3, interpolate=True)
    ax.set_ylim([-0.15, 0.4])
    ax.set_xlim([180, 320])
    ax.set_xlabel('space')
    ax.set_ylabel(r'$\langle x \rangle $')
    if np.round(0.4*times[frame_pointer[0] + 1], 2) < 10: 
        ax.set_title(f"{np.round(0.4*times[frame_pointer[0] + 1], 2)} $Jt$")
    elif np.round(0.4*times[frame_pointer[0] + 1], 2) > 10:  
        ax.set_title(f"{int(0.4*times[frame_pointer[0] + 1])} $Jt$")
    else:
        ax.set_title(f"{np.round(0.4*times[frame_pointer[0] + 1])} $Jt$")
    ax.set_aspect(150.0) 
    # Accelerate the animation
    frame_pointer[0] += 2 + 2 * int(i // 100)
    
    inset_ax.plot(0.4*np.array(times[:frame_pointer[0]+1]), total_x_mag[:frame_pointer[0]+1])
    inset_ax.plot(0.4*np.array(times[frame_pointer[0]+1]), total_x_mag[frame_pointer[0]+1], 'o', color='r')
    inset_ax.set_title("Total $x$ - magnetization", fontsize=12)
    inset_ax.set_xscale('log')
    inset_ax.set_xlim([0.1,1000])
    inset_ax.set_ylim([-2,5])
    inset_ax.set_xlabel('time [J]')
    inset_ax.axhline(y=0.0, color='k', linestyle='-')
    inset_ax.tick_params(axis='both', which='major', labelsize=10)
    return ax,

ani = animation.FuncAnimation(fig, animate_one_at_a_time, frames=number_of_slices, interval=50)

writer = animation.PillowWriter(fps=20) 
ani.save("x-magnetizaiton.gif", writer=writer)

plt.show()

total_x_mag = []
for t in range(len(time_and_space_resolved_x_magnetization)):
    total_x_mag.append(np.real(np.sum(time_and_space_resolved_x_magnetization[t])))

plt.plot(times,total_x_mag)
plt.xscale('log')
plt.show()

writer = animation.PillowWriter(fps=20) 
ani.save("x-magnetizaiton.gif", writer=writer)
