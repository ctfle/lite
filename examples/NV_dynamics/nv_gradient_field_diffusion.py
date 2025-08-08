import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import time
import pickle
import local_information as li
import matplotlib.pyplot as plt


disorder=False
seed=1

np.random.seed(seed=seed)

min_l = 3
max_l = min_l +2
L=501
dissipation_strength=0.0
delta_phi = 0.1#0.15#0.05

r_trunc_list =[0.0,0.05,0.2,0.5,1.0,0.25,0.1,0.2]
r_truncation = r_trunc_list[2]
 
num_polarized_sites=31
J = -0.1
disorder_strength =0.3
mean_dis=5.0
dis= np.random.normal(loc=0.0,scale=disorder_strength*abs(J),size=L)


# A potential decaying as 1/(n -n0)**3
def d_phi(x,del_phi,mean_dis,n0,a,r_trunc):
    if abs(a/(x-n0-mean_dis)**3 ) - del_phi < r_trunc and abs(x-n0) > mean_dis and x > n0:
            return abs(a/(x-n0-mean_dis)**3)-del_phi
    elif abs(-a/(x-n0+mean_dis)**3 ) -del_phi < r_trunc and abs(x-n0) > mean_dis and x < n0:
            return abs(a/(x-n0+mean_dis)**3)-del_phi
    else:
        return r_trunc


if disorder:
    J_list = [-J+dis[j] for j in range(L)]
    J_zz = [2*(J+dis[j]) for j in range(L)]
    x_list =[np.pi/2*d_phi(x,delta_phi,mean_dis+0.0000001,L//2, 1.0,r_truncation) for x in range(L)]
    hamiltonian_couplings = [['zz', J_zz], ['xx', J_list], ['yy', J_list], ['x',x_list]]
else:
    J_list = [-J for j in range(L)]
    J_zz = [2*(J) for j in range(L)]
    x_list = [np.pi/2*d_phi(x,delta_phi,mean_dis+0.0000001,L//2, 1.0, r_truncation) for x in range(L)]
    hamiltonian_couplings = [['zz', J_zz], ['xx', J_list], ['yy', J_list], ['x',x_list]]

# lindblad part 
plus_list = [dissipation_strength if j==L//2 else 0.0 for j in range(L)]
minus_list = [dissipation_strength if j==L//2 else 0.0 for j in range(L)]
z_list = [dissipation_strength if j==L//2 else 0.0 for j in range(L)]
jump_couplings = [['+',plus_list],['-',minus_list], ['z',z_list]]
lindbladian = li.operators.Lindbladian(max_l, hamiltonian_couplings,jump_couplings)



# Initial state - product state of single site density matrices with uniform polarization
num_polarized_sites=21
prob1x = 0.7
prob2x = 0.3
prob_up =  prob1x - prob2x 

polarized_site = np.array([[0.5, 0.5 * prob_up], [0.5 * prob_up, 0.5]])
infinite_temperature_site = np.array([[0.5, 0.0], [0.0, 0.5]])
bulk = [polarized_site for _ in range(num_polarized_sites)] 
boundary = [infinite_temperature_site for _ in range(2)]

initial_state = li.State.build_asymptotic(bulk, boundary, 1)


# Additional observables: x-magnetisation
x = [1.0 for j in range(L)]
x_list = [["x", [1.0 for j in range(L)]]]
x_mag = li.operators.Operator(x_list)

def x_magnetization(rho):
    return x_mag.expectation_value(rho)


# directory to store the data
checkpoint_folder = f"./data/2_NV_diffusion={dissipation_strength}_J={J}_L={L}_rtrunc={r_truncation}_mean_dis={mean_dis}_delta_phi={delta_phi}_num_pol={num_polarized_sites}"

# DataConfig and DataContainer is needed to communicate which observables to measure and checkpoint
data_config = li.DataConfig(
    observables=[x_magnetization],
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


lindbladian = li.operators.Lindbladian(
    max_l, hamiltonian_couplings, jump_couplings
)

# Define the system from an initial state, a Hamiltonian/ Lindbladian and the config's
system = li.OpenSystem(initial_state, lindbladian, config=config, data=data)

# the main time-evolution loop. Observables and state are checkpointed after each iteration
steps = 200
for i in range(steps):
    system.evolve(max_evolution_time=4.0, final_time=True)
    print(f"finished cycle {i} of {steps}")
    system.solver.step_size = 0.25

