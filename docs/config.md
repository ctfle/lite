# Configuration and Logging Infrastructure

## Configurations

This module provides a set of dataclasses and logging utilities designed to coordinate parameter configuration and structured logging for scientific simulations—especially those involving parallel execution with MPI. It manages configuration for time evolution, minimization, and integrations (Runge-Kutta), and offers robust, MPI-sensitive logging to file and console.

-----
## LoggingConfig
Controls where logs are written and their verbosity.
- **Parameters:**
    - `logging_folder` (`str`): Directory for log files.
    - `log_level_console` (`int`): Console log level (e.g. `logging.INFO`).
    - `log_level_file` (`int`): File log level (e.g. `logging.DEBUG`).

-----
## RungeKuttaConfig
Collects parameters related to the Runge-Kutta integration.
- **Parameters:**
    - `petz_map` (`str`): Type of Petz map for integration ('sqrt', 'exponential', etc.).
    - `RK_order` (`str`): Order of Runge-Kutta scheme ('45', '23', '1012').
    - `max_error` (`float`): Maximum allowable integration error. Must be > 0.
    - `step_size` (`float`): Adaptive stepsize to use during integration.
- **Validation:** Asserts that `max_error` is positive.

-----
## MinimizationConfig
Contains all parameters related to (local) information minimization.
- **Parameters:**
    - `minimization_threshold`: Info threshold to trigger minimization (0–1).
    - `minimization_tolerance`: Relative reduction required per step.
    - `conjugate_gradient_tolerance`: Tolerance for conjugate gradient convergence.
    - `conjugate_gradient_damping`: Damping parameter for conjugate gradient.
- **Validation:** Asserts each parameter is in (0, 1).

-----
## TimeEvolutionConfig {#time_evolution_config}
Main configuration for a simulation run.
- **Parameters:**
    - `max_l`, `min_l`: Integer levels for time evolution/minimization. Tim evolution operates between these two scales:
    information is minimized at level `min_l` after a minimization is triggered when the system reached `minimization_threshold` at `max_l`.
    - `update_dyn_max_l_threshold`: Precision threshold for dynamic updates of the evolution length-scale.
    - `system_size_tol`: Tolerance for checking density matrix equality. This parameter controls dynamic increases in the system size: 
    the effective system is increased when the density matrices deviate by `system_size_tol` from infinite temperature density matrices.
    - `shift`: Used for numerical stability (should be ≥ 0).
    - `save_checkpoint`: Whether to save checkpoints.
    - `checkpoint_folder`: Directory for checkpoints and logs.
    - `logging_config`: Instance of `LoggingConfig`.
    - `runge_kutta_config`: Instance of `RungeKuttaConfig`.
    - `minimization_config`: Instance of `MinimizationConfig`.
- **Validation:** Checks that thresholds make sense—and warns about settings with subtle effects.
- **Logging Setup:** Sets up the log directory and logging handlers if not already configured.
- **Serialization:**
    - `from_yaml`: Load configuration from a YAML file.
    - `to_yaml`: Save configuration to a YAML file (default location: `checkpoint_folder/config.yaml`).

### Usage Workflow
1. **Create a `TimeEvolutionConfig` instance**, either programmatically or by loading a YAML file.
2. On instantiation, configuration parameters are validated and logging is set up based on `logging_config`.

### Example: Creating and Saving a Configuration

```python
config = TimeEvolutionConfig(checkpoint_folder="my_data")
config.to_yaml()  # Saves to my_data/config.yaml
config = TimeEvolutionConfig.from_yaml("my_data/config.yaml")
```

-----
# Data handling

-----
## DataConfig {#data_config}

Holds configuration for observable computation and data collection.

### Attributes

| Attribute | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `observables` | `List` | `[]` | List of user-defined observables (callables) to be computed. |
| `info_lattice` | `bool` | `False` | Whether to compute information lattice observable. |
| `info_current` | `bool` | `False` | Whether to compute information current observable. |
| `diffusion_length` | `bool` | `True` | Whether to compute diffusion length. |
| `diffusion_const` | `bool` | `True` | Whether to compute diffusion constant. |
| `energy_distribution` | `bool` | `False` | Whether to compute energy distribution. |
| `times` | `bool` | `True` | Whether to record time observable. |
| `system_size` | `bool` | `True` | Whether to record system size observable. |

### Serialization Methods

- `to_dict()`: Converts configuration to dictionary for YAML serialization, converting any callable observables to their function names.
- `to_yaml(folder)`: Saves the configuration as a YAML file named `data_config.yaml` within the specified folder.
- `from_yaml(folder, full_module_path="")`: Static method that loads configuration from `data_config.yaml`, dynamically importing and converting observable function names back to callable functions from the provided module path.

-----
## DataContainer {#data_container}

Stores observable data and configuration, manages updates of observables from the current state, and handles checkpoint 
saving/loading in MPI environments.

### Initialization and Configuration

- Initializes with a `DataConfig` instance.
- Sets up internal empty dictionaries for default and custom observables depending on config.
- Can be constructed from YAML configuration with `from_yaml`.

### Observables Handling

- **Default Observables**: Computed using the [`DefaultObservables`](@ref default_observables) helper class and stored in a dictionary keyed by observable name.
- **Custom Observables**: Computed by calling user-supplied functions on `density_matrix` and stored similarly keyed by their function name.
- Supports dynamic updating/appending of data lists per observable.

### Methods
- `return_latest()`: Returns a dictionary containing the latest values (last elements) for all observables, both default
- and custom. Returns `None` if no data exists.
- `save_checkpoint(folder, warning=True)`:
Saves current observables' data to pickle files per observable name in the given folder, along with saving the 
- configuration YAML. Only rank 0 performs saving; others reset their buffers.
- `load_checkpoint(folder)`:
Loads observable data from pickle files if they exist.
- `attach_to_existing_file(folder)`:
Loads existing data, appends new data, and saves updated checkpoint to disk.
- `filepath_exists(folder)`:
Checks if the default observable files exist in a folder.
- `load_times(folder)`:
Static method to load times array from a `times.pkl` pickle.

  
## DefaultObservables {§default_observables}

Helper class that encapsulates calculation of default observables.

### Supported Observables and Their Computation Methods

| Observable | Method Name | Description |
| :-- | :-- | :-- |
| `diffusion_const` | `get_diffusion_const` | Computes diffusion constant from the given density matrix, state, and operator. |
| `diffusion_length` | `get_diffusion_length` | Computes diffusion length from `rho_dict` and operator. |
| `energy_distribution` | `get_energy_distribution` | Computes energy distribution similarly. |
| `system_size` | `get_system_size` | Returns current system size from state. |
| `times` | `get_time` | Returns current time from state. |
| `info_lattice` | `get_information_lattice` | Returns the information lattice (passed from outside). |
| `info_current` | `get_information_current` | Returns the information current from the state and operator. |

### Observable Computation Flow

- Invoked with observable name, `density_matrix`, information dictionary, `State` instance, and `SystemOperator`.
- Only performs computation on MPI rank 0; others return `None`.
- Returns the computed observable value or `None` if the observable is not recognized.
