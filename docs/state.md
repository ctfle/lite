# State {#state}

## Overview

Represents the state of the system at every point in time. Manages density matrices and boundary context. The `State` class centralizes the system’s density matrices at all spatial and hierarchical levels, providing methods for initialization, state management, boundary handling, information analysis, and checkpointing.

### State Construction Patterns

The `State` class provides various class methods to build states from assorted input forms:

- `build_finite`: Construct from a sequence of local density matrices for a finite system. To use this demands that all sites (defined in [Hamiltonian](operators.md)) are covered.
- `build_asymptotic`: Construct from separated `bulk` and (repeating) `boundary` density matrices. Not all sites need to be covered explciitly. `boundary` density matrices are assumed to cover the remaining parts of the system.
- `build`: Generic builder that dispatches to the appropriate method based on input format.

> All builders enforce consistency checks (dimensionalities, level overhead), and ensure the construction of a consistent structure (in terms of local density matrices).

-----
## Saving and Loading Checkpoints

- `save_checkpoint(folder)` saves:
    - The density matrix dictionary as a pickle file (`state.pkl`).
    - The meta data as a YAML file (`state_meta_data.yaml`).
- `from_checkpoint(folder)` loads both, restoring state.

-----
## Properties and Methods

### Properties

- `total_information`: Calculates information across all levels by summing information metrics.
- `total_information_at_dyn_max_l`: Information content at the current level.

### State Creation

- `build_finite(density_matrices, level_overhead)`
Build state for a finite system with possibly extra levels of overhead.
- `build_asymptotic(bulk_density_matrices, boundary_density_matrices, level_overhead)`
Build a state for semi-infinite or periodic structures with repeated boundary elements.
- `build(density_matrix_list, level_overhead)`
Dispatch to finite/asymptotic construction as required by input.

### Checkpointing

- `save_checkpoint(folder)`
Save state density matrices and minimal metadata in `folder` as a pickle (`state.pkl`) and YAML (`state_meta_data.yaml`).
- `from_checkpoint(folder)`
Restore the state from a previous checkpoint.

### Hierarchy and Information

- `get_information_lattice()`
Compute the entire information lattice from current `dyn_max_l` down to level 0.
- `get_information_current(operator)`
Evaluate the information current (flow) induced by a supplied operator/hamiltonian.
- `reduce_to_level(level, pop_boundary=False)`
Coarse-grain the state down to a given hierarchical level.

### System Manipulation

- `enlarge_left(nr_of_sites)`, `enlarge_right(nr_of_sites)`
Dynamically add sites at either end of the lattice (with correct hierarchical and boundary treatment).
- `check_convergence(sites_to_check_left, sites_to_check_right, tolerance)`
Iteratively test if the edges/boundaries converge to a known equilibrium (e.g., infinite temperature).
- `norm_difference(n_value, end)`
Compute the (L2) norm difference of a boundary site’s density matrix to the canonical equilibrium.
- `update_dyn_max_l(threshold, nr_of_updates)`
Dynamically add level if any information measure exceeds a specified threshold.

### Utility

- `__str__()`
Human-friendly representation of the State’s main features.
- `__eq__(other)`
Structural and information-equivalence test for two State instances.

-----
## Example Usage

```python
from local_information.state.state import State

# Build a finite state from a list of density matrices (as numpy arrays)
initial_state = State.build_finite(density_matrices=[rho1, rho2, rho3], level_overhead=1)

# Evolve the state, compute information lattice
info_lattice, _ = initial_state.get_information_lattice()

# Save to disk
initial_state.save_checkpoint("checkpoints/")

# Restore from disk
restored = State.from_checkpoint("checkpoints/")
```

