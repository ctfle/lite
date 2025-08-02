# Operator

Encapsulates a quantum operator, its couplings, and the ability to auto-generate block Hamiltonians at different locations and length scales. Tracks operator range, system size, optional disorder, and provides various convenience methods for operator algebra and property introspection.

### Initialization

```python
Operator(operator_couplings: Coupling, name: Optional[str]=None)
```

- **operator_couplings**: List describing the operator’s local terms and strengths. Each element is a tuple/list like `([“zz”], [J, J, ...])` or `([“x”], h)`. Can accommodate various interaction ranges and disorder.
- **name**: Optional label for easy identification.

-----
## Methods

- `expectation_value(rho_dict: LatticeDict) -> (LatticeDict, float)`: Computes the expectation value for the operator with respect to a dictionary of density matrices. 
Returns: A dictionary with local expectation values and the total (summed) expectation value.
- `operator` (property): Generates a `LatticeDict` containing all local operator blocks (CSR matrices) at the appropriate positions and levels. Handled via a cached property for efficiency.
- `__add__`, `__sub__`: Define addition and subtraction between Operators, producing new ones that combine the respective couplings.
- `__mul__`, `__rmul__`: Enables multiplication by a scalar. Supports syntax like `3.0 * Operator` and updates both sparse matrices and couplings.
- `__eq__`: Compares Operators by their coupling definitions.

-----
## Example Usage

```python
from local_information.operator_module import Operator, construct_operator_dict

# Define a Heisenberg Hamiltonian
couplings = [
    ['zz', [J, J, J, J]],
    ['xx', [J, J, J, J]],
    ['yy', [J, J, J, J]],
    ['z',  [h, h, h, h]],
]
op = Operator(couplings, name="Heisenberg")

# Compute expectation value w.r.t. a supplied density matrix LatticeDict
expt_dict, total_expt = op.expectation_value(rho_dict)
```
-----
## Additional functions provided

### Hamiltonian Construction Utilities

These functions automate the creation and assembly of Hamiltonians at all relevant levels/hierarchies:

- `construct_operator_dict(...)`: Constructs a dictionary (`LatticeDict`) of Hamiltonian matrices for multiple levels, using a specified coupling structure.
- `construct_hamiltonian(...)`: Builds a single Hamiltonian using QuSpin given the coupling list, length (`n_max - n_min`), and start position.
- `get_couplings(...)`: Adapts external coupling format to that required by QuSpin, supporting both homogeneous and inhomogeneous interactions.
- `compute_HH_commutator(...)`: Given a collection of Hamiltonians and a reference key (location, level), computes all commutators of the collection of Hamiltonians with the local Hamiltonian at the given key.
- `compute_H_onsite_operator_commutator(...)`: Computes the commutator of the Hamiltonian with a single-site operator.
- `add_spins(operator, number, orientation)`: Enlarges an operator by adding identity operators (`I`) on the left or right, growing its Hilbert space appropriately (via Kronecker products). Used for alignment of operators and Hilbert spaces.

### Lindbladian Construction Utilities

Functions for open-system quantum systems, where you may have local or structured dissipators ("Lindblad operators"):

- `check_lindbladian(jump_couplings, allowed_strings)`: Validates the Lindbladian (dissipative) terms, checks the range/type, and identifies disorder.
- `construct_lindbladian_dict(jump_couplings, ...)`: Returns a dictionary mapping locations and levels to the onsite Lindblad operator(s) that act there. Useful for building the full dissipation superoperator.
- `construct_lindbladian_id(n_max, jump_couplings, n_min)`: For a specified block, returns a site-indexed list of Lindblad jump operator specifications (type and value).
- `setup_onsite_L_operators(max_l, range_, type_list)`: Generates all possible one-site Lindblad operators, projected into extended Hilbert spaces as CSR matrices. These include Pauli (`x`, `y`, `z`), raising/lowering (`+`, `-`), and identity (`1`).

-----
# Hamiltonian {#hamiltonian}

The `Hamiltonian` class extends the generic `Operator` class to represent a Hamiltonian constructed from local 
couplings, organized as `LatticeDict` structures holding subsystem Hamiltonians. These data structures support 
hierarchical decompositions up to a maximum correlation length `max_l`.

### Initialization

```python
Hamiltonian(max_l: int, hamiltonian_couplings: Coupling)
```

- **max_l**: The maximum hierarchical correlation length (level). All local Hamiltonians are constructed up to (including) `max_l + range_of_couplings` so that
local time-evoultion can be performed up to level `max_l`.
- **hamiltonian_couplings**: Input list of local Hamiltonian couplings describing the system.

### Attributes

- `max_l`: Maximum hierarchical level (correlation length) local operators are available for local time-evolution.
- `hamiltonian_couplings`: Input Hamiltonian operator couplings.
- `subsystem_hamiltonian`: `LatticeDict` with all relevant subsystem Hamiltonians.
- `disorder`: Boolean indicating the presence of disorder in couplings.

> Note: The full Hamiltonian of the system can be reconstructed by summing values in the `operator` property. 
> The `subsystem_hamiltonian` attribute contains overlapping subsystem Hamiltonians, which do **not** sum up to the full
> Hamiltonian due to overlap.

### Properties
- `energy_current`: A cached property that computes the commutator of the Hamiltonian with each local operator element of the Hamiltonian itself. It returns a dictionary mapping keys (location and level) to commutator `LatticeDict`s.

### Methods

- `operator_current(operator: Operator) -> dict[LatticeDict]`: Computes the commutator of the Hamiltonian with *each element* of a given **onsite** operator.
- `save_checkpoint(folder: str)`: Saves the Hamiltonian state to the specified folder.
- `from_checkpoint(folder: str) -> Hamiltonian`: Class method to load a Hamiltonian object from a checkpoint directory.
- `__eq__`: Two Hamiltonians are equal if their `operator` arrays are identical (same matrix elements) and their `max_l` values match.

## Usage Example

```python
from local_information.operators.hamiltonian import Hamiltonian

# Suppose hamiltonian_couplings is your coupling list
max_l = 5

# Create Hamiltonian instance
hamiltonian = Hamiltonian(max_l=max_l, hamiltonian_couplings=hamiltonian_couplings)

# Access energy current (commutators of Ham with itself)
energy_currents = hamiltonian.energy_current

# Save to checkpoint
hamiltonian.save_checkpoint("saved_hamiltonian")

# Load from checkpoint
loaded_hamiltonian = hamiltonian.from_checkpoint("saved_hamiltonian")
```

-----
# Lindbladian {#lindbladian}

The `Lindbladian` class extends the generic `Operator` class to represent an **open quantum system’s Lindbladian operator**, incorporating both Hamiltonian and dissipative (jump) terms. It organizes the Lindbladian into hierarchical local subsystems stored as `LatticeDict` objects.

### Initialization

```python
Lindbladian(max_l: int, hamiltonian_couplings: Coupling, jump_couplings: Coupling)
```

- **max_l**: Maximum correlation length. All local Lindbladians are built up to level `max_l + range_of_couplings`.
- **hamiltonian_couplings**: List of coupling terms defining the Hamiltonian part of the Lindbladian.
- **jump_couplings**: List of on-site jump operator couplings describing dissipation.

### Attributes

- `max_l`: Maximum hierarchical level (correlation length).
- `hamiltonian_couplings`: Input Hamiltonian operator couplings.
- `jump_couplings`: Jump operator (dissipators) couplings.
- `subsystem_hamiltonian`: `LatticeDict` with all relevant subsystem Hamiltonians.
- `lindbladian_dict`: `LatticeDict` giving information on where to apply local Lindblad operators.
- `L_operators`: `LatticeDict` of all basic onsite Lindblad operators projected to correct Hilbert spaces.
- `disorder`: Boolean indicating the presence of disorder in couplings.

> Note: The full Lindbladian of the system can be reconstructed by summing values in the `operator` property. 
> The `subsystem_hamiltonian` attribute contains overlapping subsystem Hamiltonians, which do **not** sum up to the full
> Hamiltonian due to overlap.

### Properties

- `energy_current`: A cached property that computes the commutator of the Hamiltonian with each local operator element of the Hamiltonian itself. It returns a dictionary mapping keys (location and level) to commutator `LatticeDict`s.

### Methods

- `operator_current(operator: Operator) -> dict[LatticeDict]`: Computes commutators of the Lindbladian’s Hamiltonian with each local element of an onsite operator.
- `energy_current` (cached property): Computes the commutator of the Lindbladian’s Hamiltonian with each of its own local Hamiltonian elements.
- `save_checkpoint(folder: str)`: Saves the Lindbladian’s couplings and metadata to the specified folder.
- `from_checkpoint(folder: str) -> Lindbladian`: Loads a `Lindbladian` instance from a checkpoint.
- `__eq__(other)`: Checks equality with another `Lindbladian` instance. Two `Lindbladian` instances are considered equal if `operator` matrices,
`max_l` values and `L_operators` are equal


## Usage Example

```python
from local_information.operators.lindbladian import Lindbladian

# Define couplings for Hamiltonian and jump dissipators
hamiltonian_couplings = [...]  # your coupling list here
jump_couplings = [...]         # on-site jump operators

max_l = 5

# Create Lindbladian instance
lindbladian = Lindbladian(max_l=max_l, hamiltonian_couplings=hamiltonian_couplings, jump_couplings=jump_couplings)

# Save the Lindbladian to disk
lindbladian.save_checkpoint("checkpoints/lindbladian_run1")

# Load it back later
loaded_lindbladian = Lindbladian.from_checkpoint("checkpoints/lindbladian_run1")
```

