import os

# sets the number of threads used in the first place for numpy matrix operations
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import local_information as li


##### set some parameters #####
# min_l and max_l describe the min and max correlation length scales at which
# the algorithm operates: minimization of information is triggered as soon as
# the enough max_l-local information has build up.
min_l = 3
max_l = 6

# maximum system size
L = 301
dissipation_strength = 0.1
J = 1.0
hz = 0.2

# directory to store the data
checkpoint_folder = (
    f"./xx_yy_z={hz}_dissipation={dissipation_strength}_J={J}_L={L}_z_diffusion"
)
# set a TimeEvolutionConfig where all parameters relevant for time evolution are set
config = li.config.TimeEvolutionConfig(
    save_checkpoint=True, checkpoint_folder=checkpoint_folder, min_l=min_l, max_l=max_l
)

# define lists containing the couplings to set up a Hamiltonian/Lindbladian
J_list = [J for _ in range(L)]
z_list = [hz for _ in range(L)]
hamiltonian_couplings = [["xx", J_list], ["yy", J_list], ["z", z_list]]

# Lindblad part for the evolution
lindblad_jump_couplings = [dissipation_strength for _ in range(L)]
jump_couplings = [
    ["+", lindblad_jump_couplings],
    ["-", lindblad_jump_couplings],
    ["z", lindblad_jump_couplings],
]

# Define a Lindbladian
lindbladian_Hdd = li.operators.Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

# Initial state - We use a product state of single site density matrices.
# We use a state that is polarized in some finite regime and completely mixed elsewhere.
prob1x = 0.7
prob2x = 0.3
prob_up = prob1x - prob2x

polarized_site = np.array([[prob_up, 0.0], [0.0, 1 - prob_up]])
infinite_temperature_site = np.array([[0.5, 0.0], [0.0, 0.5]])

n_plus = 11
state = [polarized_site for _ in range(n_plus)]

# First element of the list describes the polarized part
# Second element is a list containing the remaining parts of the state i.e. those that
# repeat out to +/- infinity.
dens_matrix_list = [state, [infinite_temperature_site]]
initial_state = li.State.build(dens_matrix_list, 1)


z_list = [["z", [1.0 for _ in range(L)]]]
z_mag = li.operators.Operator(z_list)


# We define an observable that measures the z-magnetisation
def z_magnetization(rho):
    return z_mag.expectation_value(rho)


# DataConfig is needed to communicate which observables to measure and checkpoint
data_config = li.DataConfig(observables=[z_magnetization])
data = li.DataContainer(config=data_config)

# define the system using the initial state, a Hamiltonian/Lindbladian and the configs
system = li.OpenSystem(initial_state, lindbladian_Hdd, config=config, data=data)

# start the main time evolution loop. After each iteration all data is checkpointed and saved
steps = 50
for i in range(steps):
    system.evolve(max_evolution_time=5, final_time=True)
    print(f"finished cycle {i} of {steps}")
    system.solver.step_size = 0.25
