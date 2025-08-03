# Local Information Time Evolution (LITE)
LITE is a Python package for (approximate) quantum dynamics of one-dimensional quantum spin
systems. Check out our preprint [here](https://arxiv.org/pdf/2310.06036).
## Installation

If you use ```conda```  you can create an environment 
with all the required packages using:
```console
$ conda env create -n MY_ENV_NAME -f environment.yaml
```
Relpace ```MY_ENV_NAME``` with your name of choice. 
Activate the environment using
```console
$ conda activate MY_ENV_NAME
```
Alternatively, the package can be installed from the package root by running
the command
```console
$ pip install .
```
for editable installs and development dependencies use
```console
$ pip install -e '.[dev]'
```
For manual installation, install the below packages
+ [numpy](https://numpy.org) <1.25
+ [scipy](https://scipy.org)
+ [quspin](https://quspin.github.io/QuSpin/Installation.html)
+ [mpi4py](https://anaconda.org/conda-forge/mpi4py)
+ [cattrs](https://pypi.org/project/cattrs/)
+ [pyyaml](https://pypi.org/project/pyaml/)

## Usage
### Multiprocessing and Multithreading
To use multiprocessing run your script with
```console
$ mpirun -n MY_NUMBER_OF_PROCESSES python MY_SCRIPT.py
```
to run any script. This will start ```MY_NUMBER_OF_PROCESSES``` many processes on
your machine. If the system is large, this can be profitable since every time evolution
step requires many (independent) diagonalizations (For details check out our [recent work](https://arxiv.org/pdf/2310.06036)).

Many internal routines allow for multithreading to speed up calculations. To enable
multithreading add
```python
import os
os.environ["OMP_NUM_THREADS"] = "MY_NUMBER_OF_THREADS"
```
to the top of your python script for ``` MY_NUMBER_OF_THREADS``` threads.
Multithreading does not collide with multiprocessing.

## Read the intro and check the examples
Read [here]((https://ctfle.github.io/lite/)) how to use LITE and check out some [examples](https://github.com/ctfle/lite/tree/main/examples).
