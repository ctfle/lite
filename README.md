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
Alternatively, the package can be installed from the package root via `pip`
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

## Read the intro and check the examples
Read [here]((https://ctfle.github.io/lite/)) how to use LITE and check out some [examples](https://github.com/ctfle/lite/tree/main/examples).
