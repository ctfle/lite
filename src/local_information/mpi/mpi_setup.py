from __future__ import annotations
from sys import stdout
from mpi4py import MPI


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
NAME = MPI.Get_processor_name()

PARALLEL = True
if SIZE == 1:
    PARALLEL = False


def print_mpi(rank: int, message):
    print(f"RANK {rank}: ", message)
    stdout.flush()
