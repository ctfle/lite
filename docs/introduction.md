# Introduction
LITE allows to time evolve large scale long time evolution of quantum states 
for one dimensional spin systems. Here we give a basic introduction of the building blocks.
For details on the algorithm check out our recent [paper](https://arxiv.org/pdf/2310.06036)
## Hamiltonian and Lindbladian
### Hamiltonian
To specify a [Hamiltonian](@ref hamiltonian) we ne need to list the type of coupling we want to set up.
Here is an example:
```python
J_list = [J for j in range(L)]
z_list = [hz for j in range(L)]
hamiltonian_couplings = [['xx', J_list], ['yy', J_list], ['z', z_list]]
```
First, we specify values over a given range ```L```.
Next, we define the Hamiltonian couplings by grouping our value list with identifiers.
For, example ```'xx'``` implies a nearest neighbor coupling \f$ \sigma_j^x\sigma_{j+1}^x \f$
for each j and j*1 in the system, where \f$\sigma_j^i \f$ (\f$ i\in \{x,y,z\} \f$) are Pauli matrices.
Allowed identifiers are \f$ x ,y, z \f$ and \f$ 1 \f$. 
To define a Hamiltonian use

```python
import local_information as li

hamiltonian = li.operators.Hamiltonian(max_l=5, hamiltonian_couplings=hamiltonian_couplings)
```
The parameter ```max_l``` specifies the largest scale at which corresponding local Hamiltonian 
operators will be constructed and cached. Here ```max_l=5``` implies that all local Hamiltonians up to
(including) 6 sites are built and cached.

### Lindbladian
Similarly, we can define a [Lindbladian](@ref lindbladian) for time 
evolution in open quantum systems. This requires to specify the couplings for corresponding [jump operators](https://en.wikipedia.org/wiki/Lindbladian)
```python
dissipation_strength = 0.1 * J
# jump couplings 
plus_list = [dissipation_strength for j in range(L)]
minus_list = [dissipation_strength for j in range(L)]
z_list = [dissipation_strength for j in range(L)]
jump_couplings = [['z',z_list]], [['+',plus_list],['-',minus_list], ['z',z_list]]

# lindbaldian
lindbladian = li.operators.Lindbladian(max_l=5,
                                           hamiltonian_couplings=hamiltonian_couplings, 
                                           jump_couplings=jump_couplings)
```
The allowed tags for couplings are ```'+'```, ```'-'``` and ```'z'``` representing 
the Lindblad jump operators \f$ L_+ , L_- \f$ as well as dephasing \f$ L_z \f$.

## Initial state
States are handled by the [State](#state) class. State objects are organised in dicts, where keys are represented by 
tuples of the form \f$ (n, \ell) \f$ with values representing subsystem density matrices. Thereby, \f$ n \f$ indicates
the center of the corresponding subsystem and \f$ \ell \f$ represents its scale (i.e. the extend starting from 0 for a 
subsystem containing a single spin). LITE can evolve any State object.

To specify a particular (initial) state you can use ```State.build```
```python
import numpy as np
prob_up = 0.7
P = 10
single_particle_density_matrix = np.array([[prob_up, 0.0], [0.0, 1.0 - prob_up]])
asymptotic = np.array([[0.5, 0.0], [0.0, 0.5]])
state = [single_particle_density_matrix for _ in range(P)]

dens_matrix_list = [state, [asymptotic]] 
initial_state = li.State.build(dens_matrix_list, level=1, correlation=False)
```
In the example above, a state is set up where ```P``` sites are initialized with a single 
particle density matrix describing a state which is in $ \uparrow $ with probability 0.7 and
$ \downarrow $ with probability 0.3.

## Configurations
LITE involves a bunch of parameters for integrating the Schrödinger/Lindblad equation, the minimization for information 
and the algorithm itself. A documentation of all parameters can be found [here](config.md). Algorithmically, the most 
important parameters are $ min_l $ and $ max_l $, where $ max_l $ sets the largest scale at which time 
evolution of subsystems is performed before the information is minimized at scale $ min_l $. Default values are ``` min_l = 3 ``` and ```max_l = 5```
(typically larger values are required for high quality results).
### TimeEvolutionConfig
All configurations relevant for time-evolution are collected in the dataclass [TimeEvolutionConfig](@ref time_evolution_config). Create a 
TimeEvolutionConfig with appropriate parameters
```python
checkpoint_folder='./data'
config = li.config.TimeEvolutionConfig(save_checkpoint=True, checkpoint_folder=checkpoint_folder, min_l=4, max_l=6)
```
Another parameter worth to be mentioned here (since it can potentially strongly influence the outcome of simulations) is ```minimization_threshold``` 
controlling the frequency of information minimization. By default it is set to 
\f$ 1\times 10^{-2}\f$ and (from our empirical experience) should not be increased beyond \f$ 5\times 10^{-2}\f$ or 
dropped below \f$ 1\times 10^{-3}\f$.

Furthermore, TimeEvolutionConfig contains a RungeKuttaConfig as well as a MinimizationConfig which hold all parameters
regarding the adaptive RungeKutta solver and the minimization of minimization based on preconditioned conjugate gradient
optimisation.

## Operators, observables and data handling
The [operator](operators.md) class lets you define custom operators similar to Hamiltonian and Lindbladian (which subclass
Operator). You can define operators from couplings just like Hamiltonian
### Operators
```python
# define an operator
z_list =  [1.0 for j in range(L)]
couplings = [['z', z_list]]
z_mag = li.operators.Operator(couplings)
```

### Custom observables
Operators can be used to compute expectation values of [observables](@ref observables) . To do this we need to specify a function that take a single argument
(which will be replaced by the corresponding subsystem density matrix) and returns the observable value. For example,
for operator ```z_mag``` defined above, to obtain expectation values of ```z_mag``` we define a function
```python
# a function to compute the z-magnetization takes a single argument 
# it returns the 'expectation_value' of the argument `rho`
def z_magnetization(rho):
	return z_mag.expectation_value(rho)
```
Next, we need to add the function ```z_magnetization``` to the list of observables we want to compute during time 
evolution. This is done in [DataConfig](@ref data_config)

### DataConfig
The class [DataConfig](@ref data_config) allows to customize observables and specify the data computed during time evolution.
There are a bunch of pre-defined observables and quantities within LITE that can be computed provided the corresponding
flag is set ```True```. For example, to compute and store the information lattice for each time step as well as our
observable ```z_magnetization```defined above along with time and system size, use
```python
data_config = li.config.DataConfig(
    info_lattice=True,
    times=True,
    system_size=True,
    observables=[z_magnetization])
```
Note that ```times``` and ```system_size``` are ```True``` by default. 

### DataContainer
A ```data_config``` can now be used to generate a [DataContainer](@ref data_container)
```python
data = li.config.DataContainer(config=data_config)
```

## Closed and Open Systems
Using a ```State``` and ```Hamiltonian```/```Lindbladian```object as well as ```TimeEvolutionConfig``` 
and a ```DataContainer``` we can now generate now assemble a ```ClosedSystem```/```OpenSystem ```
```python
system = li.OpenSystem(init_state=initial_state, lindbladian=lindbladian, config=config,
                  data=data)
```
On ```system``` you can call ```evolve``` to time-evolve your state.
```python
for i in range(10):
    system.evolve(max_evolution_time=2.0, final_time=True)
```
This will evolve the state up to time \f$10 \times 2.0\f$ (this would also work
setting ```max_evolution_time=20.0``` however the above iterative version has
the advantage that data gets checkpointed after each iteration).

## Check out the examples
TL;DR check out the [examples](https://github.com/ctfle/lite/tree/main/examples).