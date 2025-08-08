## Multiprocessing and Multithreading
Time evolution via LITE naturally suggest the use of multiprocessing. The computational heavy lifting mostly relsted to diagonalisation of density matrices of subsystems can be parallelized.

To use multiprocessing run your script with
```console
$ mpirun -n MY_NUMBER_OF_PROCESSES python MY_SCRIPT.py
```
to run any script. This will start ```MY_NUMBER_OF_PROCESSES``` many processes on
your machine. If the system is large, this can be profitable since every time evolution step requires many (independent) diagonalizations (for details check out our [recent work](https://arxiv.org/pdf/2310.06036)).

Moreover, many internal routines allow for multithreading to speed up calculations. To enable multithreading add
```python
import os
os.environ["OMP_NUM_THREADS"] = "MY_NUMBER_OF_THREADS"
```
to the top of your python script for ``` MY_NUMBER_OF_THREADS``` threads.
Multithreading does not collide with multiprocessing.